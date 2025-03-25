import cv2
import torch
import numpy as np
import open3d as o3d
import mediapipe as mp

from PIL import Image
from collections import deque
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler


# NOTE: Change everything to video processing instead of image processing


# TODO: Optimize these constants for better results
HAND_WIDTH_MM = 90.0  # Average width of male hand in mm
CLOUD_Z_SCALE = 5.0
# Maximum expected distance between human thumb and index finger in mm when fully extended.
MAXIMUM_HAND_WIDTH_MM = 100.0


class ProcessHand:
    def __init__(self, camera_intrinsics: tuple) -> None:
        self.camera_intrinsics = camera_intrinsics

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize MediaPipe Hands for hand detection
        print("Loading MediaPipe Hands model...")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            static_image_mode=False,
        )

        # Initialize MiDaS for depth estimation
        print("Loading MiDaS model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device)
        self.midas.eval()

        # Load MiDaS transforms to resize and normalize input images
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.dpt_transform

        # Load SAM2 model for hand segmentation
        print("Loading SAM2 model...")
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large"
        )

        print("Loading Diffusion model...")
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.inpaint_pipeline.enable_attention_slicing()
        self.inpaint_pipeline.enable_xformers_memory_efficient_attention()
        self.inpaint_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.inpaint_pipeline.scheduler.config
        )

        self.gripper_width_buffer = deque(maxlen=100)

    """
    Create a segmentation mask over the hand using SAM2 model
    """

    def create_mask(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
        height, width = frame.shape[:2]
        # Convert image to RGB for SAM2 model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam2_predictor.set_image(frame_rgb)  # Set image for prediction

            # Convert landmarks to point prompts
            points = []
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                points.append([x, y])

            input_points = np.array(points)

            # Predict mask using point prompts
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=input_points,  # Pass the points as prompts
                point_labels=np.ones(
                    len(input_points)
                ),  # All points from hand are foreground
                multimask_output=False,  # Just get one mask
            )

            mask = masks[0].astype(np.uint8) * 255

        return mask

    """
    Transform input image to match MiDaS model requirements
    Estimate depth map using MiDaS model
    """

    def estimate_depth(self, image: np.ndarray) -> tuple:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform img for MiDaS model
        input_batch = self.transform(img).to(self.device)

        with torch.inference_mode():
            prediction = self.midas(input_batch)

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy and normalize for visualization
        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map_normalized = 255 * (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map, depth_map_normalized.astype(np.uint8)

    """
    Create a point cloud from combining depth map and segmented mask
    by back-projecting to 3D using camera intrinsics and depth values
    """

    def create_cloud(
        self, depth_map: np.ndarray, segmented_mask: np.ndarray
    ) -> np.ndarray:
        focal_x, focal_y, center_x, center_y = self.camera_intrinsics

        # Find points where we have both valid segmentation and valid depth
        v_coords, u_coords = np.where((segmented_mask > 0) & (depth_map > 0))
        z_values = depth_map[v_coords, u_coords]

        # Filter out zero depth values
        valid_indices = z_values > 0
        u_coords = u_coords[valid_indices]
        v_coords = v_coords[valid_indices]
        z_values = z_values[valid_indices]

        z_metric = z_values * HAND_WIDTH_MM * CLOUD_Z_SCALE / depth_map.max()

        # Back-project to 3D using camera intrinsics
        x_values = (u_coords - center_x) * z_metric / focal_x
        y_values = (v_coords - center_y) * z_metric / focal_y

        points = np.column_stack((x_values, y_values, z_metric))

        return points

    """
    Create hand mesh from hand landmarks
    """

    def create_mesh(self, landmarks: list, image_dims: tuple) -> np.ndarray:
        width, height = image_dims

        # Extract just z values to understand their range
        z_values = [landmark.z for landmark in landmarks.landmark]
        z_min = min(z_values)
        z_max = max(z_values)

        vertices = []
        for landmark in landmarks.landmark:
            # Scale z to same range as HAND_WIDTH_MM
            normalized_z = (landmark.z - z_min) / (z_max - z_min + 1e-6)
            scaled_z = normalized_z * HAND_WIDTH_MM

            vertices.append([landmark.x * width, landmark.y * height, scaled_z])

        # Define faces (triangles) connecting landmarks
        faces = [
            # Palm
            [0, 1, 5],
            [0, 5, 9],
            [0, 9, 13],
            [0, 13, 17],
            # Thumb
            [1, 2, 3],
            [2, 3, 4],
            # Index finger
            [5, 6, 7],
            [6, 7, 8],
            # Middle finger
            [9, 10, 11],
            [10, 11, 12],
            # Ring finger
            [13, 14, 15],
            [14, 15, 16],
            # Pinky
            [17, 18, 19],
            [18, 19, 20],
        ]

        dense_vertices = list(vertices)

        # Add interpolated vertices along finger segments
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_point = np.array(vertices[start_idx])
            end_point = np.array(vertices[end_idx])

            # Add 2 interpolated points between each connected pair
            for t in [0.25, 0.5, 0.75]:
                interp_point = start_point * (1 - t) + end_point * t
                dense_vertices.append(interp_point.tolist())

        return np.array(dense_vertices), np.array(faces)

    """
    Align hand mesh to point cloud using ICP for accurate 3D reconstruction
    """

    def icp_registration(self, cloud: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(vertices)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(cloud)

        # Calculate centroids
        source_centroid = np.mean(vertices, axis=0)
        target_centroid = np.mean(cloud, axis=0)

        # Calculate initial translation to align centroids
        initial_translation = target_centroid - source_centroid

        # Create initial transformation matrix
        trans_init = np.eye(4)
        trans_init[:3, 3] = initial_translation

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=0.025,
            init=trans_init,  # Initial transformation
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100
            ),
        )

        transformation = result.transformation

        return transformation

    """
    Refine landmarks based on the icp transformation
    """

    def refine_landmarks(
        self, landmarks: list, transform: int, image_dims: tuple
    ) -> list:
        width, height = image_dims

        # Extract z range for normalization, similar to _create_mesh
        z_values = [landmark.z for landmark in landmarks.landmark]
        z_min = min(z_values)
        z_max = max(z_values)

        refined_landmarks = []
        for landmark in landmarks.landmark:
            # Use consistent scaling with _create_mesh
            normalized_z = (landmark.z - z_min) / (z_max - z_min + 1e-6)
            scaled_z = normalized_z * HAND_WIDTH_MM

            point = np.array([landmark.x * width, landmark.y * height, scaled_z, 1.0])

            # Apply transformation to 3D point
            transformed = np.dot(transform, point)
            refined_landmarks.append(transformed[:3])

        return refined_landmarks

    # TODO: Implement better constraints that limit last joint of each finger to a single DOF
    def apply_constraints(self, landmarks: list):
        constrained = np.array(landmarks)

        # Define finger joint indices
        # MediaPipe hand model: Wrist is 0, thumb is 1-4, index is 5-8, etc.
        thumb_indices = [1, 2, 3, 4]
        index_indices = [5, 6, 7, 8]

        # Constrain the last two joints of thumb and index finger as per the paper
        for finger_indices in [thumb_indices, index_indices]:
            # Get the last three joints (two segments)
            if len(finger_indices) >= 3:
                # Get joint positions
                joint1 = constrained[finger_indices[-3]]
                joint2 = constrained[finger_indices[-2]]
                joint3 = constrained[finger_indices[-1]]

                # Direction of the first segment
                dir1 = joint2 - joint1
                dir1 = dir1 / np.linalg.norm(dir1)

                # Instead of full ball joint, constrain last joint's direction to follow previous segment
                ideal_dir = dir1.copy()
                actual_dir = joint3 - joint2
                actual_length = np.linalg.norm(actual_dir)

                # Force the direction to be within a cone of the previous segment
                # (limiting to single degree of freedom approximately)
                corrected_dir = ideal_dir * actual_length

                # Apply the correction
                constrained[finger_indices[-1]] = joint2 + corrected_dir

        return constrained

    """
    Extract robot parameters from refined landmarks:
    1. Target Position: Midpoint between thumb tip and index tip
    2. Target Orientation: Normal to the best-fitting plane of thumb and index finger
    3. Gripper Width: Distance between thumb tip and index tip
    """

    def get_robot_params(self, refined_landmarks: list) -> tuple:
        landmarks = np.array(refined_landmarks)

        # Define indices for specific parts of the hand
        thumb_indices = [1, 2, 3, 4]  # Thumb landmarks
        index_indices = [5, 6, 7, 8]  # Index finger landmarks
        thumb_tip_idx = 4
        index_tip_idx = 8

        # 1. Set target position as midpoint between thumb tip and index tip
        thumb_tip = landmarks[thumb_tip_idx]
        index_tip = landmarks[index_tip_idx]
        position = (thumb_tip + index_tip) / 2

        # 2. Calculate orientation
        # Get all thumb and index finger points for plane fitting
        thumb_points = landmarks[thumb_indices]
        index_points = landmarks[index_indices]
        finger_points = np.vstack([thumb_points, index_points])

        # Fit plane to finger points
        centroid = np.mean(finger_points, axis=0)
        centered_points = finger_points - centroid

        # Use SVD to find the normal to the best-fitting plane
        u, s, vh = np.linalg.svd(centered_points)
        # The normal is the last right singular vector
        plane_normal = vh[2, :]
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Fit a principal axis through thumb points
        # Using direction from thumb base to tip for more robustness
        thumb_direction = landmarks[thumb_tip_idx] - landmarks[thumb_indices[0]]
        thumb_axis = thumb_direction / np.linalg.norm(thumb_direction)

        # Ensure thumb_axis is orthogonal to plane_normal
        thumb_axis = thumb_axis - np.dot(thumb_axis, plane_normal) * plane_normal
        thumb_axis = thumb_axis / np.linalg.norm(thumb_axis)

        # Compute third axis as cross product to create orthogonal frame
        cross_axis = np.cross(plane_normal, thumb_axis)
        cross_axis = cross_axis / np.linalg.norm(cross_axis)

        # Create rotation matrix that aligns with the paper's description
        # z-axis as normal, x-axis along thumb direction
        rotation_matrix = np.column_stack([thumb_axis, cross_axis, plane_normal])

        # 3. Calculate gripper width
        gripper_width = np.linalg.norm(thumb_tip - index_tip)
        self.gripper_width_buffer.append(gripper_width)

        # Apply 20th percentile threshold as specified in the paper
        if len(self.gripper_width_buffer) > 5:  # Need enough samples
            min_width = np.percentile(self.gripper_width_buffer, 20)
            if gripper_width < min_width:
                gripper_width = 0.0  # Fully closed gripper when below threshold
            else:
                # Scale gripper_width to robot's specific range
                gripper_width = min(1.0, gripper_width / MAXIMUM_HAND_WIDTH_MM)

        # TODO: Convert from camera frame to robot frame
        # Note: This requires the extrinsic matrix from camera to robot
        # If extrinsics are available, uncomment and use this code:
        # if hasattr(self, "camera_to_robot_transform"):
        #     # Convert position to homogeneous coordinates
        #     pos_homogeneous = np.append(position, 1.0)
        #     # Apply transformation
        #     robot_pos_homogeneous = np.dot(self.camera_to_robot_transform, pos_homogeneous)
        #     position = robot_pos_homogeneous[:3]
        #
        #     # Convert rotation (special orthogonal transformation)
        #     rotation_in_robot_frame = np.dot(self.camera_to_robot_transform[:3, :3], rotation_matrix)
        #     rotation_matrix = rotation_in_robot_frame

        return position, rotation_matrix, gripper_width

    def inpaint_hand(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # The mask needs to be inverted for SD inpainting (255 for area to inpaint)
        inpaint_mask = mask.copy()

        pil_image = Image.fromarray(rgb_image)
        pil_mask = Image.fromarray(inpaint_mask)

        # Resize if needed (SD works best with smaller images)
        width, height = pil_image.size
        max_size = 512
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            pil_image = pil_image.resize(new_size)
            pil_mask = pil_mask.resize(new_size)

            result = self.inpaint_pipeline(
                prompt="seamless background continuation, consistent lighting and texture, natural scene",
                negative_prompt="hands, fingers, arms, human body parts, discontinuity, edge artifacts, blurriness, distortion",
                image=pil_image,
                mask_image=pil_mask,
                guidance_scale=7.5,
            ).images[0]

        if max(width, height) > max_size:
            result = result.resize((width, height))

        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        return result_np
