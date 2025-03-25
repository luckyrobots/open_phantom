import os
import cv2
import numpy as np
import pybullet as p

from utils.handle_urdf import handle_urdf
from scipy.spatial.transform import Rotation


class RobotManager:
    def __init__(self, urdf_path: str, camera_intrinsics: tuple) -> None:
        self.physics_client = p.connect(p.DIRECT)  # Headless mode

        robot_urdf = handle_urdf(urdf_path)

        parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        absolute_urdf_path = os.path.join(parent_dir, robot_urdf)

        self.robot_id = p.loadURDF(absolute_urdf_path, useFixedBase=True)
        self.joint_count = p.getNumJoints(self.robot_id)
        # TODO: Figure out a way to handle multiple end effectors
        self.end_effector_index = self._find_gripper_joints()[-1]

        _, _, center_x, center_y = camera_intrinsics
        # Set up rendering parameters
        self.img_width = int(center_x * 2)
        self.img_height = int(center_y * 2)

    def set_robot_pose(self, position, orientation, gripper_width):
        # Convert orientation matrix to quaternion
        r = Rotation.from_matrix(orientation)
        quaternion = r.as_quat()

        # Get current joint positions as seed
        joint_indices = list(range(self.joint_count))
        joint_states = p.getJointStates(self.robot_id, joint_indices)
        current_positions = [state[0] for state in joint_states]

        # Exclude gripper joints for IK calculation
        gripper_joints = self._find_gripper_joints()
        ik_joint_indices = [j for j in joint_indices if j not in gripper_joints]

        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            targetPosition=position,
            targetOrientation=quaternion,
            currentPositions=current_positions,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        # Apply best solution
        for i, joint_idx in enumerate(ik_joint_indices):
            if i < len(solution):
                p.resetJointState(self.robot_id, joint_idx, solution[i])

        self._set_gripper_width(gripper_width)

        return solution

    def _find_gripper_joints(self) -> list:
        gripper_joints = []
        gripper_keywords = [
            "gripper",
            "tool",
            "tcp",
            "end_effector",
            "hand",
            "finger",
            "claw",
            "pinch",
        ]

        for i in range(self.joint_count):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8").lower()

            if any(keyword in joint_name for keyword in gripper_keywords):
                gripper_joints.append(i)

        return gripper_joints

    def _set_gripper_width(self, width: float) -> None:
        gripper_joints = self._find_gripper_joints()

        assert gripper_joints, "No gripper joints found, cannot set gripper width"

        # Clamp width to valid range
        width = max(0.0, min(1.0, width))

        # Get joint info to determine limits
        for joint_idx in gripper_joints:
            info = p.getJointInfo(self.robot_id, joint_idx)
            lower_limit = info[8]  # Lower limit
            upper_limit = info[9]  # Upper limit

            # Calculate target position based on width
            # For some grippers, smaller values mean close and larger values mean open
            # For others, it's the opposite, so we need to check the joint info
            if (
                "left" in info[1].decode("utf-8").lower()
                or "open" in info[1].decode("utf-8").lower()
            ):
                # For left/open joints (opening movement)
                target_pos = lower_limit + width * (upper_limit - lower_limit)
            else:
                # For right/close joints (closing movement)
                target_pos = upper_limit - width * (upper_limit - lower_limit)

            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=100,  # Lower force for gripper to prevent damage
            )

    def render_robot(
        self, inpainted_frame: np.ndarray, depth_map: np.ndarray, camera_params=None
    ) -> np.ndarray:
        assert self.robot_id >= 0, "Robot not properly loaded"

        # Resize depth map if needed
        if depth_map.shape[:2] != (self.img_height, self.img_width):
            depth_map = cv2.resize(depth_map, (self.img_width, self.img_height))

        # Get current robot pose for camera targeting
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        robot_pos = link_state[0]  # Position of the end effector

        # Set up camera parameters
        if camera_params is None:
            cam_target = robot_pos
            cam_distance = 0.3  # Closer view
            cam_yaw = 0
            cam_pitch = -30
            cam_roll = 0
        else:
            cam_target, cam_distance, cam_yaw, cam_pitch, cam_roll = camera_params

        print(f"Robot position: {robot_pos}")
        print(f"Camera target: {cam_target}, distance: {cam_distance}")

        # Compute view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_distance,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=cam_roll,
            upAxisIndex=2,
        )

        # Compute projection matrix
        aspect = self.img_width / self.img_height
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=aspect, nearVal=0.01, farVal=100.0
        )

        # Render the scene
        img_arr = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Extract RGB and depth
        rgb = np.reshape(img_arr[2], (self.img_height, self.img_width, 4))
        rgb = rgb[:, :, :3]  # Remove alpha channel
        robot_depth = np.reshape(img_arr[3], (self.img_height, self.img_width))

        # Save the raw robot rendering for debugging
        cv2.imwrite("robot_debug_rgb.png", rgb)

        # Resize background if needed
        frame_h, frame_w = inpainted_frame.shape[:2]
        if frame_w != self.img_width or frame_h != self.img_height:
            frame_resized = cv2.resize(
                inpainted_frame, (self.img_width, self.img_height)
            )
        else:
            frame_resized = inpainted_frame

        # Create basic robot mask (where robot pixels are visible)
        robot_mask = (robot_depth < 0.99).astype(np.float32)

        # Save the robot mask for debugging
        cv2.imwrite("robot_debug_mask.png", (robot_mask * 255).astype(np.uint8))

        # Check if robot is visible at all
        if np.sum(robot_mask) == 0:
            print("WARNING: Robot is not visible in the rendered image!")
            # If robot not visible, return the inpainted frame
            return frame_resized

        # More straightforward compositing without occlusion for testing
        # Just overlay the robot on the frame where the robot mask is active
        final_mask = np.stack([robot_mask, robot_mask, robot_mask], axis=2)
        composite = frame_resized * (1 - final_mask) + rgb * final_mask

        return composite.astype(np.uint8)

    def __del__(self) -> None:
        if hasattr(self, "physics_client"):
            try:
                p.disconnect(self.physics_client)
            except:
                pass
