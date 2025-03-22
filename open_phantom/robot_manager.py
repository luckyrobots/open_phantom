import os

import cv2
import numpy as np
import pybullet as p
from utils.handle_urdf import handle_urdf


class RobotManager:
    def __init__(self, urdf_path: str, camera_intrinsics: tuple) -> None:
        self.physics_client = p.connect(p.DIRECT)

        robot_urdf = handle_urdf(urdf_path)
        self.robot_id = self._load_robot(robot_urdf)

        self.joint_count = p.getNumJoints(self.robot_id)
        self.end_effector_index = self._find_end_effector()

        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        # Set up rendering parameters
        self.img_width = int(self.cx * 2)
        self.img_height = int(self.cy * 2)

    # Load the robot URDF into PyBullet
    def _load_robot(self, robot_urdf: str) -> int:
        try:
            robot_id = p.loadURDF(
                robot_urdf,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
            )
        except p.error as e:
            print(f"PyBullet error when loading URDF: {e}")
            raise e

        robot_name = p.getBodyInfo(robot_id)[1].decode("utf-8")
        print(f"Successfully loaded {robot_name} robot with ID: {robot_id}")

        return robot_id

    # NOTE: Only applicable if the robot has one end effector
    def _find_end_effector(self) -> int:
        assert self.joint_count > 0, "Robot has no joints"

        # Keywords to look for in joint names to identify end effector
        keywords = ["gripper", "tool", "tcp", "end_effector", "hand"]

        for i in range(self.joint_count):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8").lower()

            # Check if any keyword is in the joint name
            if any(keyword in joint_name for keyword in keywords):
                return i

        # If no specific end effector found, use the last joint in the chain
        return self.joint_count - 1

    # TODO: Use inverse kinematics to set the robot pose
    def set_robot_pose(
        self,
        position: np.ndarray,
        orientation_vectors: np.ndarray,
        gripper_width: float,
    ) -> None:
        pass

    # Render the robot in some scene using some camera parameters
    def render_robot(self, bg_image=None, camera_params=None):
        assert self.robot_id >= 0, "Robot not properly loaded"

        # Set up camera parameters
        if camera_params is None:
            # Default camera setup
            cam_target = [0, 0, 0]
            cam_distance = 1.0
            cam_yaw = 0
            cam_pitch = -30
            cam_roll = 0
        else:
            cam_target, cam_distance, cam_yaw, cam_pitch, cam_roll = camera_params

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
        depth = np.reshape(img_arr[3], (self.img_height, self.img_width))

        # If background image is provided, composite
        if bg_image is not None:
            # Resize background if needed
            bg_h, bg_w = bg_image.shape[:2]
            if bg_w != self.img_width or bg_h != self.img_height:
                bg_resized = cv2.resize(bg_image, (self.img_width, self.img_height))
            else:
                bg_resized = bg_image

            # Create mask from depth
            mask = (depth < 0.99).astype(np.float32)
            mask = np.stack([mask, mask, mask], axis=2)

            # Composite
            composite = bg_resized * (1 - mask) + rgb * mask
            return composite.astype(np.uint8)

        return rgb.astype(np.uint8)

    def __del__(self) -> None:
        if hasattr(self, "physics_client"):
            try:
                p.disconnect(self.physics_client)
            except:
                pass


if __name__ == "__main__":
    cwd = os.getcwd()
    urdf_path = os.path.join(
        cwd,
        "notebook/phantom/urdf/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM.urdf",
    )
    camera_intrinsics = (320, 240, 320, 240)  # Random intrinsics for example

    robot_vis = RobotManager(urdf_path, camera_intrinsics)
    rendered_image = robot_vis.render_robot()

    # Option 1: Display the image using OpenCV
    cv2.imshow("Robot Render", rendered_image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

    # Option 2: Save the image to a file
    output_path = "robot_render.png"
    cv2.imwrite(output_path, rendered_image)
    print(f"Render saved to {output_path}")
