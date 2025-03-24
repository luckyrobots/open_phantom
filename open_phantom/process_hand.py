import os
import time
import cv2
import torch
import numpy as np
import open3d as o3d
import mediapipe as mp

from tqdm import tqdm
from collections import deque
from utils.visualizations import *
from sam2.sam2_image_predictor import SAM2ImagePredictor


# TODO: Optimize these constants for better results
HAND_WIDTH_MM = 90.0  # Average width of male hand in mm
CLOUD_Z_SCALE = 5.0
# Maximum expected distance between human thumb and index finger in mm when fully extended.
MAXIMUM_HAND_WIDTH_MM = 100.0


class ProcessHand:
    def __init__(self) -> None:
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

        self.gripper_width_buffer = deque(maxlen=100)

    """
    Create a segmentation mask over the hand using SAM2 model
    """

    def _create_mask(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
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

    def _estimate_depth(self, image: np.ndarray) -> tuple:
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

    def _create_cloud(
        self, depth_map: np.ndarray, segmented_mask: np.ndarray
    ) -> np.ndarray:
        focal_x, focal_y, center_x, center_y = self.camera_intrinsics

        v_coords, u_coords = np.where(segmented_mask > 0)
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

    def _create_mesh(self, landmarks: list, image_dims: tuple) -> np.ndarray:
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
            for t in [0.33, 0.66]:
                interp_point = start_point * (1 - t) + end_point * t
                dense_vertices.append(interp_point.tolist())

        return np.array(dense_vertices), np.array(faces)

    """
    Align hand mesh to point cloud using ICP for accurate 3D reconstruction
    """

    def _icp_registration(self, cloud: np.ndarray, vertices: np.ndarray) -> np.ndarray:
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
            max_correspondence_distance=0.05,
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

    def _refine_landmarks(
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
    def _apply_constraints(self, landmarks: list):
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

    def _get_robot_params(self, refined_landmarks: list) -> tuple:
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

        # Convert from camera frame to robot frame
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

    def record_video(self) -> str:
        output_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_filename = os.path.join(output_dir, f"recorded_video_{timestamp}.mp4")

        cap = cv2.VideoCapture(0)

        assert cap.isOpened(), "Failed to open camera."

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Update camera intrinsics based on video dimensions
        self.camera_intrinsics = (width * 0.8, height * 0.8, width / 2, height / 2)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        recording = False
        did_record = False

        print("Camera is active. Press 'r' to start/stop recording, 'q' to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            assert success, "Failed to read from camera."

            # Mirror the image for more intuitive viewing
            frame = cv2.flip(frame, 1)

            # Create a separate display frame for showing the recording indicator
            display_frame = frame.copy()

            # Display recording indicator ONLY on the display frame
            if recording:
                cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(
                    display_frame,
                    "RECORDING",
                    (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Show the display frame (with indicator if recording)
            cv2.imshow("Video Recording", display_frame)

            # Write the original frame (without indicator) to video file if recording
            if recording and out is not None:
                out.write(frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                recording = not recording
                if recording:
                    print(f"Started recording to {video_filename}")
                    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                else:
                    if out is not None:
                        out.release()
                        print(f"Stopped recording. Video saved to {video_filename}")
                    did_record = True
                    break

        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()

        return video_filename if did_record else None

    def process_video(self, video_path: str) -> tuple:
        assert video_path, "Video path is required."

        cap = cv2.VideoCapture(video_path)

        assert cap.isOpened(), f"Failed to open video file {video_path}"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Update camera intrinsics based on video dimensions
        self.camera_intrinsics = (width * 0.8, height * 0.8, width / 2, height / 2)

        base_path = os.path.splitext(video_path)[0]
        segmented_output_path = f"{base_path}_masked.mp4"
        depth_output_path = f"{base_path}_depth.mp4"
        mesh_output_path = f"{base_path}_mesh.mp4"
        registration_output_path = f"{base_path}_registration.mp4"
        constraints_output_path = f"{base_path}_constraints.mp4"
        robot_output_path = f"{base_path}_robot.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        segmented_out = cv2.VideoWriter(
            segmented_output_path, fourcc, fps, (width, height)
        )
        depth_out = cv2.VideoWriter(depth_output_path, fourcc, fps, (width, height))
        mesh_out = cv2.VideoWriter(mesh_output_path, fourcc, fps, (width, height))
        reg_out = cv2.VideoWriter(
            registration_output_path, fourcc, fps, (640, 480)
        )  # Fixed size
        constraints_out = cv2.VideoWriter(
            constraints_output_path, fourcc, fps, (width, height)
        )
        robot_out = cv2.VideoWriter(robot_output_path, fourcc, fps, (width, height))

        print(f"Processing video with {total_frames} frames...")
        for _ in tqdm(range(total_frames)):
            success, frame = cap.read()
            assert success, "Failed to read frame from video."

            # Convert image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe Hands
            results = self.hands.process(image_rgb)

            # Initialize output frames
            segmented_frame = frame.copy()
            depth_frame = np.zeros((height, width, 3), dtype=np.uint8)
            mesh_frame = frame.copy()
            reg_frame = (
                np.ones((480, 640, 3), dtype=np.uint8) * 255
            )  # Fixed size, white background
            constraints_frame = frame.copy()
            robot_frame = frame.copy()

            # Process if hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    segmented_mask = self._create_mask(frame, hand_landmarks)
                    mask_overlay = frame.copy()
                    mask_overlay[segmented_mask > 0] = [0, 0, 255]  # Red color for mask
                    segmented_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

                    depth_map, depth_vis = self._estimate_depth(frame)
                    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    depth_frame = depth_colored.copy()

                    cloud = self._create_cloud(depth_map, segmented_mask)

                    hand_vertices, hand_faces = self._create_mesh(
                        hand_landmarks, (width, height)
                    )
                    mesh_frame = visualize_mesh(frame, hand_vertices, hand_faces)

                    icp_transform = self._icp_registration(cloud, hand_vertices)
                    reg_frame = visualize_registration(
                        cloud, hand_vertices, icp_transform
                    )

                    refined_landmarks = self._refine_landmarks(
                        hand_landmarks, icp_transform, (width, height)
                    )

                    original_refined = refined_landmarks.copy()

                    constrained_landmarks = self._apply_constraints(refined_landmarks)
                    constraints_frame = visualize_constraints(
                        frame,
                        original_refined,
                        constrained_landmarks,
                        self.camera_intrinsics,
                    )

                    position, orientation, gripper_width = self._get_robot_params(
                        constrained_landmarks
                    )
                    robot_frame = visualize_robot_params(
                        frame,
                        position,
                        orientation,
                        gripper_width,
                        self.camera_intrinsics,
                    )

            segmented_out.write(segmented_frame)
            depth_out.write(depth_frame)
            mesh_out.write(mesh_frame)
            reg_out.write(reg_frame)
            constraints_out.write(constraints_frame)
            robot_out.write(robot_frame)

            display_scale = 0.5
            display_size = (int(width * display_scale), int(height * display_scale))
            reg_display_size = (int(640 * display_scale), int(480 * display_scale))

            cv2.imshow("Segmented", cv2.resize(segmented_frame, display_size))
            cv2.imshow("Depth", cv2.resize(depth_frame, display_size))
            cv2.imshow("Mesh", cv2.resize(mesh_frame, display_size))
            cv2.imshow("Registration", cv2.resize(reg_frame, reg_display_size))
            cv2.imshow("Constraints", cv2.resize(constraints_frame, display_size))
            cv2.imshow("Robot Parameters", cv2.resize(robot_frame, display_size))

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        cap.release()
        segmented_out.release()
        depth_out.release()
        mesh_out.release()
        reg_out.release()
        constraints_out.release()
        robot_out.release()
        cv2.destroyAllWindows()

        print(f"Processing complete. Results saved to:")
        print(f"- Hand mask: {segmented_output_path}")
        print(f"- Depth visualization: {depth_output_path}")
        print(f"- Mesh visualization: {mesh_output_path}")
        print(f"- Registration visualization: {registration_output_path}")
        print(f"- Constraints visualization: {constraints_output_path}")
        print(f"- Robot parameters: {robot_output_path}")

        return {
            "segmented": segmented_output_path,
            "depth": depth_output_path,
            "mesh": mesh_output_path,
            "registration": registration_output_path,
            "constraints": constraints_output_path,
            "robot": robot_output_path,
        }
