import os
import re
import cv2
import torch
import numpy as np
import mediapipe as mp

from collections import deque
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from intrinsic import calibrate_intrinsics


REFERENCE_OBJECT_SIZE = 0.50  # 50cm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def robot_to_screen_pos(robot_pos: list, K: np.ndarray) -> tuple:
    """
    Project 3D robot position to 2D screen position using intrinsic matrix
    """
    # Create homogeneous point
    robot_pt_homogeneous = np.append(robot_pos, 1.0)

    # Project using intrinsic matrix
    screen_pt = np.dot(K, robot_pt_homogeneous[:3])

    # Normalize by z coordinate to get pixel coordinates
    screen_x, screen_y = screen_pt[:2] / screen_pt[2]

    # Convert to integers
    screen_x, screen_y = int(screen_x), int(screen_y)

    return screen_x, screen_y


# Filter hand position using Savitzky-Golay to smooth out noise
def filter_hand_position(position: list, history: deque, window_size: int = 7) -> list:
    history.append(position)
    if len(history) < 5:
        return position

    history_array = np.array(history)

    if len(history) >= window_size:
        filtered_x = savgol_filter(history_array[:, 0], window_size, 2)
        filtered_y = savgol_filter(history_array[:, 1], window_size, 2)
        filtered_z = savgol_filter(history_array[:, 2], window_size, 2)
        filtered_position = np.array([filtered_x[-1], filtered_y[-1], filtered_z[-1]])
    else:
        # Simple average if not enough history
        filtered_position = np.mean(history_array, axis=0)

    return filtered_position


# Return 3D hand position from MediaPipe landmarks using intrinsic matrix
def get_hand_pos(
    landmarks: list, intrinsic_matrix: np.ndarray, width: int, height: int
) -> np.ndarray:
    """
    Get 3D hand position using camera intrinsics for proper projection
    """
    # Use thumb tip and index finger tip
    thumb_tip = landmarks.landmark[4]  # Thumb tip index
    index_tip = landmarks.landmark[8]  # Index finger tip index

    # Convert landmark coordinates to pixel coordinates
    thumb_img = np.array([thumb_tip.x * width, thumb_tip.y * height, 1.0])
    index_img = np.array([index_tip.x * width, index_tip.y * height, 1.0])

    # Apply inverse of intrinsic matrix to get normalized camera coordinates
    K_inv = np.linalg.inv(intrinsic_matrix)
    thumb_norm = np.dot(K_inv, thumb_img)
    index_norm = np.dot(K_inv, index_img)

    # Scale by depth (using MediaPipe's relative depth estimate)
    # Note: This scaling is approximate, ideally would use a proper depth sensor
    thumb_depth = max(0.1, thumb_tip.z * 100)  # Avoid negative/zero depth
    index_depth = max(0.1, index_tip.z * 100)  # Avoid negative/zero depth

    thumb_pos = thumb_norm * thumb_depth
    index_pos = index_norm * index_depth

    # Midpoint between thumb and index finger
    position = (thumb_pos + index_pos) / 2

    return position[:3]  # Return x, y, z


def measure_pinch_size(
    landmarks: list,
    intrinsic_matrix: np.ndarray,
    frame: np.ndarray,
    midas_model,
    midas_transform,
    width: int,
    height: int,
) -> float:
    """
    Measure distance between thumb and index finger in 3D space
    using intrinsics and MiDaS depth estimation
    """
    # Use thumb tip and index finger tip
    thumb_tip = landmarks.landmark[4]  # Thumb tip
    index_tip = landmarks.landmark[8]  # Index finger tip

    # Convert to pixel coordinates
    thumb_pixel = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    index_pixel = (int(index_tip.x * width), int(index_tip.y * height))

    # Get depth map using MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = midas_transform(img).to(DEVICE)

    with torch.no_grad():
        depth_map = midas_model(img_input)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = depth_map.cpu().numpy()

    # Sample depths at thumb and index finger locations
    thumb_depth = depth_map[thumb_pixel[1], thumb_pixel[0]]
    index_depth = depth_map[index_pixel[1], index_pixel[0]]

    # Normalize depths (MiDaS gives relative depths)
    # We need to convert to metric scale
    # TODO: This scaling factor needs calibration
    depth_scale = 5.0
    thumb_depth_metric = thumb_depth * depth_scale
    index_depth_metric = index_depth * depth_scale

    # Apply inverse of intrinsic matrix to get normalized camera coordinates
    K_inv = np.linalg.inv(intrinsic_matrix)

    # Convert to homogeneous coordinates
    thumb_img = np.array([thumb_pixel[0], thumb_pixel[1], 1.0])
    index_img = np.array([index_pixel[0], index_pixel[1], 1.0])

    # Get normalized camera coordinates
    thumb_norm = np.dot(K_inv, thumb_img)
    index_norm = np.dot(K_inv, index_img)

    # Scale by depth to get 3D coordinates
    thumb_pos = thumb_norm * thumb_depth_metric
    index_pos = index_norm * index_depth_metric

    # Calculate Euclidean distance
    distance = np.linalg.norm(thumb_pos - index_pos)

    # Visualize depths for debugging
    cv2.putText(
        frame,
        f"Thumb depth: {thumb_depth:.2f}, Index depth: {index_depth:.2f}",
        (10, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    return distance


def is_hand_stable(
    position, history: deque, n_frames: int = 50, threshold: float = 8.0
) -> bool:
    """Check if hand position is stable using a simple threshold"""
    history.append(position)

    if len(history) < n_frames:
        return False

    positions = np.array(history)
    max_movement = np.max(np.linalg.norm(positions - positions[-1], axis=1))

    return max_movement < threshold


# Compute extrinsics using PnP (Perspective-n-Point)
def compute_extrinsics_pnp(
    camera_points: np.ndarray, robot_points: np.ndarray, intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute camera extrinsics using PnP algorithm.

    Args:
        camera_points: 3D points in camera frame
        robot_points: 3D points in robot frame
        intrinsic_matrix: Camera intrinsic matrix

    Returns:
        4x4 transformation matrix from camera to robot frame
    """
    # Convert 3D camera points to 2D image points using intrinsic matrix
    image_points = []
    for pt in camera_points:
        # Project 3D point to 2D using the camera model
        pt_homogeneous = np.append(pt, 1.0)
        img_pt = np.dot(intrinsic_matrix, pt_homogeneous[:3])
        # Normalize to get image coordinates
        img_pt = img_pt[:2] / img_pt[2]
        image_points.append(img_pt)

    image_points = np.array(image_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)

    # Use OpenCV's solvePnP to get camera pose relative to robot
    success, rvec, tvec = cv2.solvePnP(
        robot_points,
        image_points,
        intrinsic_matrix,
        distCoeffs=None,  # Add distortion coefficients if available
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    # Convert rotation vector to rotation matrix
    rot_matrix, _ = cv2.Rodrigues(rvec)

    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = tvec.flatten()

    return transform


def refine_calibration(
    camera_points: np.ndarray,
    robot_points: np.ndarray,
    initial_transform: np.ndarray,
    intrinsic_matrix: np.ndarray,
) -> np.ndarray:
    """
    Refine calibration using bundle adjustment optimization

    Args:
        camera_points: 3D points in camera frame
        robot_points: 3D points in robot frame
        initial_transform: Initial transformation matrix from camera to robot
        intrinsic_matrix: Camera intrinsic matrix

    Returns:
        Refined 4x4 transformation matrix
    """
    # Extract rotation and translation from initial transform
    R = initial_transform[:3, :3]
    t = initial_transform[:3, 3]

    # Convert rotation matrix to Rodriguez vector
    rvec, _ = cv2.Rodrigues(R)

    # Initial parameters (rotation vector and translation)
    params = np.concatenate([rvec.flatten(), t])

    # Define the error function
    def error_function(params):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)

        errors = []
        for robot_pt, camera_pt in zip(robot_points, camera_points):
            robot_pt = robot_pt.reshape(1, 3)

            # Project robot point to camera using the current parameters
            img_pt, _ = cv2.projectPoints(robot_pt, rvec, tvec, intrinsic_matrix, None)
            img_pt = img_pt.flatten()

            # Convert camera point to image using intrinsics
            camera_pt_h = np.append(camera_pt, 1.0)
            proj_camera_pt = np.dot(intrinsic_matrix, camera_pt_h[:3])
            proj_camera_pt = proj_camera_pt[:2] / proj_camera_pt[2]

            # Calculate reprojection error
            error = img_pt - proj_camera_pt
            errors.extend(error)

        return np.array(errors)

    # Run optimization
    result = least_squares(error_function, params, method="lm")

    # Convert optimized parameters back to transformation matrix
    rvec_opt = result.x[:3].reshape(3, 1)
    tvec_opt = result.x[3:].reshape(3, 1)

    R_opt, _ = cv2.Rodrigues(rvec_opt)

    transform_opt = np.eye(4)
    transform_opt[:3, :3] = R_opt
    transform_opt[:3, 3] = tvec_opt.flatten()

    return transform_opt


def calc_reprojection_error(
    robot_points: list,
    camera_points: list,
    transform: np.ndarray,
    intrinsic_matrix: np.ndarray,
) -> tuple:
    """
    Calculate reprojection error using the intrinsic matrix for proper projection
    """
    # Extract rotation and translation
    R = transform[:3, :3]
    t = transform[:3, 3]

    # Convert to rotation vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Project robot points to image space
    robot_points = np.array(robot_points, dtype=np.float32)
    projected_pts, _ = cv2.projectPoints(
        robot_points, rvec, tvec, intrinsic_matrix, None
    )
    projected_pts = projected_pts.reshape(-1, 2)

    # Convert camera points to image space
    camera_img_pts = []
    for pt in camera_points:
        pt_h = np.append(pt, 1.0)
        img_pt = np.dot(intrinsic_matrix, pt_h[:3])
        img_pt = img_pt[:2] / img_pt[2]
        camera_img_pts.append(img_pt)

    camera_img_pts = np.array(camera_img_pts)

    # Calculate errors in image space (pixel distances)
    errors = np.linalg.norm(projected_pts - camera_img_pts, axis=1)

    return errors, np.mean(errors)


def get_intrinsics() -> np.ndarray:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(file_dir)

    filename = "intrinsics.npz"
    if filename in files:
        try:
            calibration_data = np.load(os.path.join(file_dir, filename))
            intrinsics = calibration_data["K"]
            print("Intrinsic matrix loaded.")
            print(intrinsics)
            return intrinsics
        except (KeyError, FileNotFoundError) as e:
            print(f"Error loading intrinsic matrix: {e}")

    print("Intrinsic matrix not found or invalid.")
    print("Calibrating intrinsic matrix...")
    intrinsics = calibrate_intrinsics()
    print("Intrinsic matrix calibrated.")

    return intrinsics


def calibrate_extrinsics() -> np.ndarray:
    # Load intrinsic matrix
    intrinsics = get_intrinsics()
    if intrinsics is None:
        print(
            "Failed to load or calibrate intrinsic matrix. Cannot proceed with extrinsic calibration."
        )
        return None

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        static_image_mode=False,
    )

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(DEVICE)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transform.small_transform

    # All in robot coordinates
    reference_positions = [
        [0.3, 0.0, 0.4],  # Center
        [0.3, 0.2, 0.4],  # Right
        [0.3, -0.2, 0.4],  # Left
        [0.3, 0.0, 0.6],  # Higher
        [0.3, 0.0, 0.2],  # Lower
        [0.2, 0.2, 0.3],  # Front right bottom
        [0.2, -0.2, 0.3],  # Front left bottom
        [0.2, 0.2, 0.5],  # Front right top
        [0.2, -0.2, 0.5],  # Front left top
    ]

    # Validation positions for reprojection error
    validation_positions = [[0.25, 0.1, 0.35], [0.35, -0.1, 0.45], [0.3, 0.15, 0.5]]

    cap = cv2.VideoCapture(0)

    success, frame = cap.read()
    assert success, "Failed to access camera"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = 50
    position_history = deque(maxlen=2 * n_frames)

    print("Step 1: Scale calibration")
    print(
        f"Hold your thumb and index finger exactly {REFERENCE_OBJECT_SIZE*100}cm apart"
    )

    scale_factor = None
    scale_calibration_complete = False

    reference_distances = []

    while not scale_calibration_complete:
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        cv2.putText(
            frame,
            f"Hold thumb and index finger exactly {REFERENCE_OBJECT_SIZE*100}cm apart",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Hold position steady...",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            distance = measure_pinch_size(hand_landmarks, intrinsics, width, height)

            cv2.putText(
                frame,
                f"Current distance: {distance:.2f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            hand_position = get_hand_pos(hand_landmarks, intrinsics, width, height)
            if is_hand_stable(hand_position, position_history, n_frames=30):
                reference_distances.append(distance)
                cv2.putText(
                    frame,
                    "Measurement taken!",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Take 5 measurements for robustness
                if len(reference_distances) >= 5:
                    # Calculate scale factor (median to avoid outliers)
                    median_distance = np.median(reference_distances)
                    scale_factor = REFERENCE_OBJECT_SIZE / median_distance
                    print(f"Scale factor: {scale_factor}")
                    scale_calibration_complete = True
                    position_history.clear()
                else:
                    # Wait a bit between measurements
                    for i in range(30):
                        cv2.imshow("Calibration", frame)
                        cv2.waitKey(100)
                    position_history.clear()

        cv2.putText(
            frame,
            f"Measurements: {len(reference_distances)}/5",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Step 2: Position calibration")
    print("Please move your hand to the highlighted positions on screen.")

    current_target = 0
    camera_positions = []
    filtering_history = deque(maxlen=n_frames)

    calibration_complete = False

    while not calibration_complete:
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        target_position = reference_positions[current_target]
        # Use intrinsic matrix for accurate projection
        screen_target = robot_to_screen_pos(target_position, intrinsics)

        # Draw target
        cv2.circle(frame, screen_target, 50, (0, 255, 0), 2)
        cv2.circle(frame, screen_target, 10, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Target {current_target+1}: Move hand here",
            (screen_target[0] - 100, screen_target[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        hand_position = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_hand_position = get_hand_pos(hand_landmarks, intrinsics, width, height)

            hand_position = filter_hand_position(raw_hand_position, filtering_history)

            hand_position_scaled = hand_position * scale_factor

            # Project hand position to screen
            hand_pt_h = np.append(hand_position, 1.0)
            screen_pt = np.dot(intrinsics, hand_pt_h[:3])
            screen_pt = (
                int(screen_pt[0] / screen_pt[2]),
                int(screen_pt[1] / screen_pt[2]),
            )

            # Draw hand position indicator
            cv2.circle(frame, screen_pt, 15, (255, 0, 0), -1)

            # Check if hand is close to target (in 2D screen space for simplicity)
            distance = np.sqrt(
                (screen_pt[0] - screen_target[0]) ** 2
                + (screen_pt[1] - screen_target[1]) ** 2
            )

            if distance < 50:  # Threshold in pixels
                cv2.putText(
                    frame,
                    "Hold position steady...",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                if is_hand_stable(hand_position, position_history, n_frames):
                    camera_positions.append(hand_position_scaled)
                    print(f"Position {current_target+1} recorded!")

                    # Move to next target and reset history
                    current_target += 1
                    position_history.clear()

                    if current_target >= len(reference_positions):
                        calibration_complete = True
                    else:
                        print(f"Please move to position {current_target+1}")

        cv2.putText(
            frame,
            f"Calibrating: {current_target}/{len(reference_positions)} positions",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not calibration_complete:
        print("Calibration aborted.")
        cap.release()
        cv2.destroyAllWindows()
        return None

    # Use PnP for initial extrinsic estimation
    robot_points = np.array(reference_positions)
    camera_points = np.array(camera_positions)

    # Compute initial transform using PnP
    initial_transform = compute_extrinsics_pnp(camera_points, robot_points, intrinsics)
    print("Initial transform computed using PnP:")
    print(initial_transform)

    # Refine using bundle adjustment
    refined_transform = refine_calibration(
        camera_points, robot_points, initial_transform, intrinsics
    )
    print("Refined transform after bundle adjustment:")
    print(refined_transform)

    print("Calibration complete! Now validating...")

    validation_camera_positions = []
    current_target = 0
    position_history.clear()

    while current_target < len(validation_positions):
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        target_position = validation_positions[current_target]
        screen_target = robot_to_screen_pos(target_position, intrinsics)

        # Draw validation target
        cv2.circle(frame, screen_target, 50, (255, 165, 0), 2)  # Orange
        cv2.circle(frame, screen_target, 10, (255, 165, 0), -1)
        cv2.putText(
            frame,
            f"Validation {current_target+1}: Move hand here",
            (screen_target[0] - 100, screen_target[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 165, 0),
            2,
        )

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            raw_hand_position = get_hand_pos(hand_landmarks, intrinsics, width, height)
            hand_position = filter_hand_position(raw_hand_position, filtering_history)
            hand_position_scaled = hand_position * scale_factor

            # Project hand position to screen for visualization
            hand_pt_h = np.append(hand_position, 1.0)
            screen_pt = np.dot(intrinsics, hand_pt_h[:3])
            screen_pt = (
                int(screen_pt[0] / screen_pt[2]),
                int(screen_pt[1] / screen_pt[2]),
            )

            # Draw hand position indicator
            cv2.circle(frame, screen_pt, 15, (255, 0, 0), -1)

            # Check if hand is close to target
            distance = np.sqrt(
                (screen_pt[0] - screen_target[0]) ** 2
                + (screen_pt[1] - screen_target[1]) ** 2
            )

            if distance < 50:
                cv2.putText(
                    frame,
                    "Hold position steady for validation...",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 165, 0),
                    2,
                )

                if is_hand_stable(hand_position, position_history, n_frames=30):
                    validation_camera_positions.append(hand_position_scaled)
                    print(f"Validation position {current_target+1} recorded!")
                    current_target += 1
                    position_history.clear()

        cv2.putText(
            frame,
            f"Validation: {current_target}/{len(validation_positions)} positions",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Use the refined transform for validation
    errors, mean_error = calc_reprojection_error(
        validation_positions, validation_camera_positions, refined_transform, intrinsics
    )

    print("Validation complete!")
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    print(f"Max reprojection error: {np.max(errors):.4f} pixels")
    print(f"Min reprojection error: {np.min(errors):.4f} pixels")

    calibration_data = {
        "transform": refined_transform,
        "scale_factor": scale_factor,
        "reprojection_error": mean_error,
        "intrinsic_matrix": intrinsics,
        "calibration_date": np.datetime64("now"),
    }
    np.save("extrinsics.npy", calibration_data)

    # Show validation results visually
    results_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(
        results_img,
        "Calibration Results",
        (150, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        results_img,
        f"Scale Factor: {scale_factor:.6f}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        results_img,
        f"Mean Reprojection Error: {mean_error:.4f} pixels",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0)
        if mean_error < 10
        else (0, 165, 255)
        if mean_error < 20
        else (0, 0, 255),
        2,
    )
    cv2.putText(
        results_img,
        f"Max Reprojection Error: {np.max(errors):.4f} pixels",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0)
        if np.max(errors) < 15
        else (0, 165, 255)
        if np.max(errors) < 30
        else (0, 0, 255),
        2,
    )

    quality = (
        "Excellent"
        if mean_error < 5
        else "Good"
        if mean_error < 10
        else "Acceptable"
        if mean_error < 20
        else "Poor"
    )
    cv2.putText(
        results_img,
        f"Calibration Quality: {quality}",
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0)
        if quality in ["Excellent", "Good"]
        else (0, 165, 255)
        if quality == "Acceptable"
        else (0, 0, 255),
        2,
    )

    cv2.imshow("Calibration Results", results_img)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    return refined_transform


if __name__ == "__main__":
    transform = calibrate_extrinsics()

    if transform is not None:
        print("Calibration successful!")
        print(transform)
    else:
        print("Calibration failed.")
