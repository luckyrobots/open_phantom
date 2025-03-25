import cv2
import numpy as np
import mediapipe as mp

from collections import deque


# Convert 3D robot positions to 2D screen positions
def robot_to_screen_pos(robot_pos: list, width: int, height: int) -> tuple:
    center_x, center_y = width // 2, height // 2
    scale_factor = min(width, height) / 2

    # Y-axis maps to horizontal
    screen_x = int(center_x + robot_pos[1] * scale_factor)
    # Z-axis maps to vertical (inverted)
    screen_y = int(center_y - robot_pos[2] * scale_factor)

    return screen_x, screen_y


# Return 3D hand position from MediaPipe landmarks
def get_hand_pos(landmarks, width: int, height: int):
    # Use thumb tip and index finger tip
    thumb_tip = landmarks.landmark[4]  # Thumb tip index
    index_tip = landmarks.landmark[8]  # Index finger tip index

    # Convert to 3D position
    thumb_pos = np.array(
        [thumb_tip.x * width, thumb_tip.y * height, thumb_tip.z * 100]
    )  # Rough scaling
    index_pos = np.array([index_tip.x * width, index_tip.y * height, index_tip.z * 100])

    # Midpoint between thumb and index finger
    position = (thumb_pos + index_pos) / 2

    return position


# Check if hand position is stable at current location
def is_hand_stable(
    position, history: deque, frames: int = 50, threshold: float = 8.0
) -> bool:
    history.append(position)

    if len(history) < frames:
        return False

    positions = np.array(history)
    max_movement = np.max(np.linalg.norm(positions - positions[-1], axis=1))

    return max_movement < threshold


def calibrate_camera() -> np.ndarray:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        static_image_mode=False,
    )

    # Robot frame positions
    reference_positions = [
        [0.3, 0.0, 0.4],  # Center
        [0.3, 0.2, 0.4],  # Right
        [0.3, -0.2, 0.4],  # Left
        [0.3, 0.0, 0.6],  # Higher
    ]

    cap = cv2.VideoCapture(0)

    # Get camera dimensions
    success, frame = cap.read()
    assert success, "Failed to access camera"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    position_history = deque(maxlen=15)

    current_target = 0
    camera_positions = []
    calibration_complete = False

    print("Automatic calibration starting.")
    print("Please move your hand to the highlighted positions on screen.")

    while not calibration_complete:
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        target_position = reference_positions[current_target]
        screen_target = robot_to_screen_pos(target_position, width, height)

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
            hand_landmarks = results.multi_hand_landmarks[0]  # Just use the first hand

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_position = get_hand_pos(hand_landmarks)

            # Draw hand position indicator
            hand_x, hand_y = int(hand_position[0]), int(hand_position[1])
            cv2.circle(frame, (hand_x, hand_y), 15, (255, 0, 0), -1)

            # Check if hand is close to target (in 2D screen space for simplicity)
            distance = np.sqrt(
                (hand_x - screen_target[0]) ** 2 + (hand_y - screen_target[1]) ** 2
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

                if is_hand_stable(hand_position, position_history):
                    camera_positions.append(hand_position)
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

    cap.release()
    cv2.destroyAllWindows()

    if not calibration_complete:
        print("Calibration aborted.")
        return None

    robot_points = np.array(reference_positions)
    robot_centroid = np.mean(robot_points, axis=0)
    robot_centered = robot_points - robot_centroid

    camera_points = np.array(camera_positions)
    camera_centroid = np.mean(camera_points, axis=0)
    camera_centered = camera_points - camera_centroid

    # Find rotation using SVD
    H = np.dot(camera_centered.T, robot_centered)
    U, S, Vt = np.linalg.svd(H)
    rotation = np.dot(U, Vt)

    # Check that it's a proper rotation matrix
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(U, Vt)

    translation = robot_centroid - np.dot(rotation, camera_centroid)

    # Create homogeneous transform matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    print("Calibration complete!")

    np.save("camera_extrinsics.npy", transform)

    return transform


if __name__ == "__main__":
    transform = calibrate_camera()
    if transform is not None:
        print("Transform matrix:")
        print(transform)
        print("Saved to camera_to_robot_transform.npy")
