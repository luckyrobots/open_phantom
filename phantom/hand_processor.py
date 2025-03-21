import os
import cv2
import time
import torch
import numpy as np
import open3d as o3d
import mediapipe as mp

from tqdm import tqdm
from utils.visualizations import *
from phantom.robot_manager import RobotManager
# from diffusers import StableDiffusionInpaintPipeline


class HandProcessor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            static_image_mode=False,
        )
        
        # NOTE: Look into better depth estimation models
        # Initialize MiDaS for depth estimation
        print("Loading MiDaS model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load MiDaS transforms to resize and normalize input images
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.dpt_transform
    
    # Segment hand from image using MediaPipe Hands landmarks
    def _create_mask(self, image: np.ndarray, landmarks: list, thickness: int=5, padding:int=10) -> np.ndarray:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks.landmark]
        
        # Draw filled circles at each landmark
        for point in points:
            cv2.circle(mask, point, thickness, 255, -1)
                
        # Connect all landmarks with thick lines
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(mask, points[start_idx], points[end_idx], 255, thickness)
        
        # Create palm by connecting base of fingers with wrist
        palm_points = [points[0], points[1], points[5], points[9], points[13], points[17]]
        cv2.fillPoly(mask, [np.array(palm_points)], 255)
        
        # Create shape between fingers
        finger_bases = [(1,5), (5,9), (9,13), (13,17)]
        for base1, base2 in finger_bases:
            triangle = np.array([points[0], points[base1], points[base2]])
            cv2.fillPoly(mask, [triangle], 255)
        
        # Dilate to smooth and expand slightly
        kernel = np.ones((padding, padding), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        
        return mask

    """
    Transform input image to match MiDaS model requirements
    Estimate depth map using MiDaS model
    """
    # TODO: Look into why screen goes black sometimes
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
        
        # Convert to numpy and normalize to 0-255 for visualization
        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map_normalized = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map, depth_map_normalized.astype(np.uint8)
    
    # TODO: Look into better depth scaling
    def _create_cloud(self, depth_map: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
        focal_x, focal_y, center_x, center_y = self.camera_intrinsics
        
        v_coords, u_coords = np.where(hand_mask > 0)        
        z_values = depth_map[v_coords, u_coords]
        
        # Filter out zero depth values
        valid_indices = z_values > 0
        u_coords = u_coords[valid_indices]
        v_coords = v_coords[valid_indices]
        z_values = z_values[valid_indices]
        
        # NOTE: Abritrary scaling factor for depth
        z_metric = z_values * 0.5
        
        # Back-project to 3D using camera intrinsics
        x_values = (u_coords - center_x) * z_metric / focal_x
        y_values = (v_coords - center_y) * z_metric / focal_y
        
        points = np.column_stack((x_values, y_values, z_metric))
        
        return points
    
    # TODO: Look into better Z scaling
    def _create_mesh(self, landmarks: list, image_dims: tuple) -> np.ndarray:
        width, height = image_dims
        
        vertices = []
        for landmark in landmarks.landmark:
            vertices.append([
                landmark.x * width,
                landmark.y * height,
                landmark.z * width
            ])
            
        # Define faces (triangles) connecting landmarks
        faces = [
            # Palm
            [0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17],
            # Thumb
            [1, 2, 3], [2, 3, 4],
            # Index finger
            [5, 6, 7], [6, 7, 8],
            # Middle finger
            [9, 10, 11], [10, 11, 12],
            # Ring finger
            [13, 14, 15], [14, 15, 16],
            # Pinky
            [17, 18, 19], [18, 19, 20]
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
                interp_point = start_point * (1-t) + end_point * t
                dense_vertices.append(interp_point.tolist())
        
        return np.array(dense_vertices), np.array(faces)
    
    def _icp_registration(self, cloud: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        # Normalize both point clouds to handle scale differences
        cloud_centroid = np.mean(cloud, axis=0)
        vertices_centroid = np.mean(vertices, axis=0)
        
        cloud_centered = cloud - cloud_centroid
        vertices_centered = vertices - vertices_centroid
        
        cloud_scale = np.max(np.linalg.norm(cloud_centered, axis=1))
        vertices_scale = np.max(np.linalg.norm(vertices_centered, axis=1))
        
        cloud_normalized = cloud_centered / cloud_scale
        vertices_normalized = vertices_centered / vertices_scale
        
        # Create Open3D point clouds
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(vertices_normalized)
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(cloud_normalized)
        
        # Optional: Downsample for better performance
        source = source.voxel_down_sample(voxel_size=0.01)
        target = target.voxel_down_sample(voxel_size=0.01)
        
        # Initial identity transformation
        trans_init = np.eye(4)
        
        # Run ICP with better parameters
        result = o3d.pipelines.registration.registration_icp(
            source,
            target, 
            max_correspondence_distance=0.2,  # Increased for better matching
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        # Create denormalization transformation
        denorm_transform = np.eye(4)
        denorm_transform[:3, :3] *= (cloud_scale / vertices_scale)
        
        # Combine transformations
        transform_to_origin = np.eye(4)
        transform_to_origin[:3, 3] = -vertices_centroid
        
        transform_from_origin = np.eye(4)
        transform_from_origin[:3, 3] = cloud_centroid
        
        final_transform = np.matmul(transform_from_origin, 
                                np.matmul(denorm_transform, 
                                            np.matmul(result.transformation, transform_to_origin)))
        
        return final_transform
    
    def _refine_landmarks(self, landmarks: list, transform: int, image_dims: tuple):
        width, height = image_dims
        
        refined_landmarks = []
        for landmark in landmarks.landmark:
            point = np.array([
                landmark.x * width,
                landmark.y * height,
                landmark.z * width, # TODO: Figure out better scaling factor
                1.0
            ])
            
            # Apply transformation to 3D point
            transformed = np.dot(transform, point)
            refined_landmarks.append(transformed[:3])
            
        return refined_landmarks
    
    def _apply_constraints(self, landmarks: list):
        constrained = np.array(landmarks)
        
        # Define finger joint indices
        # MediaPipe hand model: Wrist is 0, thumb is 1-4, index is 5-8, etc.
        thumb_indices = [1, 2, 3, 4]
        index_indices = [5, 6, 7, 8]
        
        # Constrain the last two joints of thumb and index finger
        # as mentioned in the paper
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
                
                # Instead of full ball joint, constrain the last joint's direction
                # to follow similar direction as the previous segment
                ideal_dir = dir1.copy()
                actual_dir = joint3 - joint2
                actual_length = np.linalg.norm(actual_dir)
                
                # Force the direction to be within a cone of the previous segment
                # (limiting to single degree of freedom approximately)
                corrected_dir = ideal_dir * actual_length
                
                # Apply the correction
                constrained[finger_indices[-1]] = joint2 + corrected_dir
        
        return constrained
    
    def _get_robot_params(self, refined_landmarks) -> tuple:
        # Extract key landmarks
        wrist = refined_landmarks[0]  # Wrist landmark
        thumb_tip = refined_landmarks[4]  # Thumb tip
        index_tip = refined_landmarks[8]  # Index finger tip
        
        # Calculate end effector position (midpoint between thumb and index tips)
        position = (thumb_tip + index_tip) / 2
        
        # Calculate vectors for orientation
        v1 = thumb_tip - wrist  # Vector from wrist to thumb tip
        v2 = index_tip - wrist  # Vector from wrist to index tip
        
        # Calculate normal to hand plane
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        else:
            # Default if vectors are collinear
            normal = np.array([0, 0, 1])
        
        # Calculate main direction along thumb
        direction = thumb_tip - wrist
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            # Default if thumb is at wrist (unlikely)
            direction = np.array([1, 0, 0])
        
        # Calculate gripper width
        gripper_width = np.linalg.norm(thumb_tip - index_tip)
        
        return position, direction, normal, gripper_width
    
    def record_video(self) -> str:
        output_dir = os.path.join(os.getcwd(), 'recordings')
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                cv2.putText(display_frame, "RECORDING", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show the display frame (with indicator if recording)
            cv2.imshow('Video Recording', display_frame)
            
            # Write the original frame (without indicator) to video file if recording
            if recording and out is not None:
                out.write(frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # NOTE: Haven't tested what happens if user records multiple videos in one session (probably overwrites?)
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print(f"Started recording to {video_filename}")
                    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                else:
                    if out is not None:
                        out.release()
                        print(f"Stopped recording. Video saved to {video_filename}")
                    did_record = True
        
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        segmented_out = cv2.VideoWriter(segmented_output_path, fourcc, fps, (width, height))
        depth_out = cv2.VideoWriter(depth_output_path, fourcc, fps, (width, height))
        mesh_out = cv2.VideoWriter(mesh_output_path, fourcc, fps, (width, height))
        reg_out = cv2.VideoWriter(registration_output_path, fourcc, fps, (640, 480))  # Fixed size
        constraints_out = cv2.VideoWriter(constraints_output_path, fourcc, fps, (width, height))
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
            reg_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Fixed size, white background
            constraints_frame = frame.copy()
            robot_frame = frame.copy()
            
            # Process if hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Segment hand
                    hand_mask = self._create_mask(frame, hand_landmarks)
                    mask_overlay = frame.copy()
                    mask_overlay[hand_mask > 0] = [0, 0, 255]  # Red color for mask
                    segmented_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
                    
                    # Depth estimation
                    depth_map, depth_vis = self._estimate_depth(frame)
                    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    depth_frame = depth_colored.copy()
                    
                    # Create hand mesh
                    hand_vertices, hand_faces = self._create_mesh(hand_landmarks, (width, height))
                    mesh_frame = visualize_mesh(frame, hand_vertices, hand_faces)
                    
                    # Create point cloud from depth
                    cloud = self._create_cloud(depth_map, hand_mask)
                    
                    # Perform ICP registration
                    icp_transform = self._icp_registration(cloud, hand_vertices)
                    reg_frame = visualize_registration(cloud, hand_vertices, icp_transform)
                    
                    # Refine landmarks using ICP transformation
                    refined_landmarks = self._refine_landmarks(hand_landmarks, icp_transform, (width, height))
                    
                    # Store pre-constraint landmarks for visualization
                    original_refined = refined_landmarks.copy()
                    
                    # Apply anatomical constraints
                    constrained_landmarks = self._apply_constraints(refined_landmarks)
                    constraints_frame = visualize_constraints(frame, original_refined, constrained_landmarks, self.camera_intrinsics)
                    
                    # # Calculate robot parameters
                    # position, direction, normal, gripper_width = self._get_robot_params(constrained_landmarks)
                    # robot_frame = visualize_robot_params(frame, position, direction, normal, gripper_width)
            
            segmented_out.write(segmented_frame)
            depth_out.write(depth_frame)
            mesh_out.write(mesh_frame)
            reg_out.write(reg_frame)
            constraints_out.write(constraints_frame)
            robot_out.write(robot_frame)
            
            display_scale = 0.5
            display_size = (int(width * display_scale), int(height * display_scale))
            reg_display_size = (int(640 * display_scale), int(480 * display_scale))
            
            cv2.imshow('Segmented', cv2.resize(segmented_frame, display_size))
            cv2.imshow('Depth', cv2.resize(depth_frame, display_size))
            cv2.imshow('Mesh', cv2.resize(mesh_frame, display_size))
            cv2.imshow('Registration', cv2.resize(reg_frame, reg_display_size))
            cv2.imshow('Constraints', cv2.resize(constraints_frame, display_size))
            cv2.imshow('Robot Parameters', cv2.resize(robot_frame, display_size))
            
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
            "robot": robot_output_path
        }