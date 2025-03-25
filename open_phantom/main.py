import os
import cv2
import time
import numpy as np

from tqdm import tqdm
from utils.visualizations import *
from process_hand import ProcessHand
from robot_manager import RobotManager


def record_video() -> str | None:
    output_dir = os.path.join(os.getcwd(), "recordings")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_filename = os.path.join(output_dir, f"recorded_video_{timestamp}.mp4")

    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), "Failed to open camera."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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


def process_video(video_path: str, urdf_path: str) -> None:
    assert video_path, "Video path is required."

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), f"Failed to open video file {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Update camera intrinsics based on video dimensions
    camera_intrinsics = (width * 0.8, height * 0.8, width / 2, height / 2)

    processor = ProcessHand(camera_intrinsics)
    robot_manager = RobotManager(urdf_path, camera_intrinsics)

    base_path = os.path.splitext(video_path)[0]
    segmented_output_path = f"{base_path}_masked.mp4"
    depth_output_path = f"{base_path}_depth.mp4"
    mesh_output_path = f"{base_path}_mesh.mp4"
    registration_output_path = f"{base_path}_registration.mp4"
    constraints_output_path = f"{base_path}_constraints.mp4"
    robot_params_output_path = f"{base_path}_robot_params.mp4"
    inpainted_output_path = f"{base_path}_inpainted.mp4"
    robot_overlay_output_path = f"{base_path}_robot_overlay.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    segmented_out = cv2.VideoWriter(segmented_output_path, fourcc, fps, (width, height))
    depth_out = cv2.VideoWriter(depth_output_path, fourcc, fps, (width, height))
    mesh_out = cv2.VideoWriter(mesh_output_path, fourcc, fps, (width, height))
    cloud_out = cv2.VideoWriter(f"{base_path}_cloud.mp4", fourcc, fps, (width, height))
    reg_out = cv2.VideoWriter(registration_output_path, fourcc, fps, (width, height))
    constraints_out = cv2.VideoWriter(
        constraints_output_path, fourcc, fps, (width, height)
    )
    robot_params_out = cv2.VideoWriter(
        robot_params_output_path, fourcc, fps, (width, height)
    )
    inpainted_out = cv2.VideoWriter(inpainted_output_path, fourcc, fps, (width, height))
    robot_overlay_out = cv2.VideoWriter(
        robot_overlay_output_path, fourcc, fps, (width, height)
    )

    print(f"Processing video with {total_frames} frames...")
    for _ in tqdm(range(total_frames)):
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from video.")
            break

        # Convert image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = processor.hands.process(image_rgb)

        # Initialize output frames
        segmented_frame = frame.copy()
        depth_frame = np.zeros((height, width, 3), dtype=np.uint8)
        mesh_frame = frame.copy()
        cloud_frame = (
            np.ones((480, 640, 3), dtype=np.uint8) * 255
        )  # Fixed size, white background
        reg_frame = (
            np.ones((480, 640, 3), dtype=np.uint8) * 255
        )  # Fixed size, white background
        constraints_frame = frame.copy()
        robot_params_frame = frame.copy()
        inpainted_frame = frame.copy()
        robot_overlay_frame = frame.copy()

        # Process if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_mask = processor.create_mask(frame, hand_landmarks)
                mask_overlay = frame.copy()
                mask_overlay[hand_mask > 0] = [0, 0, 255]  # Red color for mask
                segmented_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

                depth_map, depth_vis = processor.estimate_depth(frame)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_frame = depth_colored.copy()

                cloud = processor.create_cloud(depth_map, hand_mask)
                cloud_frame = visualize_cloud(cloud, (width, height))

                hand_vertices, hand_faces = processor.create_mesh(
                    hand_landmarks, (width, height)
                )
                mesh_frame = visualize_mesh(frame, hand_vertices, hand_faces)

                icp_transform = processor.icp_registration(cloud, hand_vertices)
                reg_frame = visualize_registration(cloud, hand_vertices, icp_transform)

                refined_landmarks = processor.refine_landmarks(
                    hand_landmarks, icp_transform, (width, height)
                )

                original_refined = refined_landmarks.copy()

                constrained_landmarks = processor.apply_constraints(refined_landmarks)
                constraints_frame = visualize_constraints(
                    frame,
                    original_refined,
                    constrained_landmarks,
                    camera_intrinsics,
                )

                position, orientation, gripper_width = processor.get_robot_params(
                    constrained_landmarks
                )
                robot_params_frame = visualize_robot_params(
                    frame,
                    position,
                    orientation,
                    gripper_width,
                    camera_intrinsics,
                )

                # inpainted_frame = processor.inpaint_hand(frame, hand_mask)

                robot_manager.set_robot_pose(position, orientation, gripper_width)
                robot_overlay_frame = robot_manager.render_robot(frame, depth_map)

        segmented_out.write(segmented_frame)
        depth_out.write(depth_frame)
        mesh_out.write(mesh_frame)
        cloud_out.write(cloud_frame)
        reg_out.write(reg_frame)
        constraints_out.write(constraints_frame)
        robot_params_out.write(robot_params_frame)
        # inpainted_out.write(inpainted_frame)
        robot_overlay_out.write(robot_overlay_frame)

        display_scale = 0.5
        display_size = (int(width * display_scale), int(height * display_scale))
        reg_display_size = (int(640 * display_scale), int(480 * display_scale))

        cv2.imshow("Segmented", cv2.resize(segmented_frame, display_size))
        cv2.imshow("Depth", cv2.resize(depth_frame, display_size))
        cv2.imshow("Mesh", cv2.resize(mesh_frame, display_size))
        cv2.imshow("Cloud", cv2.resize(cloud_frame, display_size))
        cv2.imshow("Registration", cv2.resize(reg_frame, reg_display_size))
        cv2.imshow("Constraints", cv2.resize(constraints_frame, display_size))
        cv2.imshow("Robot Parameters", cv2.resize(robot_params_frame, display_size))
        # cv2.imshow("Inpainted", cv2.resize(inpainted_frame, display_size))
        cv2.imshow("Robot Overlay", cv2.resize(robot_overlay_frame, display_size))

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    segmented_out.release()
    depth_out.release()
    mesh_out.release()
    cloud_out.release()
    reg_out.release()
    constraints_out.release()
    robot_params_out.release()
    inpainted_out.release()
    robot_overlay_out.release()
    cv2.destroyAllWindows()


def main(debug: bool = True) -> None:
    if debug:
        video_path = "recordings/recorded_video_20250325-141723.mp4"
        urdf_path = (
            "urdf/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM_fixed.urdf"
        )
        process_video(video_path, urdf_path)
        return

    record_option = input("Record a new video? (y/n): ")

    if record_option.lower() == "y":
        print("Starting video recording...")
        video_path = record_video()
        if not video_path:
            print("Video recording failed.")
            return

        print(f"Video recorded successfully to {video_path}")
        process_option = input("Process the video now? (y/n): ")
        if process_option.lower() == "y":
            urdf_path = input("Enter the path to a robot URDF file: ")
            while not os.path.exists(urdf_path):
                print("Error: URDF file not found.")
                urdf_path = input("Enter the path to a robot URDF file: ")
        else:
            print("Video processing skipped.")
            return
    else:
        video_path = input("Enter the path to a video file: ")
        while not os.path.exists(video_path):
            print("Error: Video file not found.")
            video_path = input("Enter the path to a video file: ")

        urdf_path = input("Enter the path to a robot URDF file: ")
        while not os.path.exists(urdf_path):
            print("Error: URDF file not found.")
            urdf_path = input("Enter the path to a robot URDF file: ")

    process_video(video_path, urdf_path)


if __name__ == "__main__":
    main()
