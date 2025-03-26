import os
import re
import cv2
import time
import numpy as np

from datetime import datetime
from calibration_math import *


class PointCollector:
    """Collect points for vanishing point calibration"""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image.copy()
        self.display_image = image.copy()

        self.height, self.width = image.shape[:2]

        self.points = []
        self.line_sets = []
        self.n_line_sets = 3
        self.current_points = []
        self.colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        self.window_name = "Camera Calibration"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: None, param: None
    ) -> None:
        """Mouse callback function for collecting points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_points) < 4:
                self.current_points.append((x, y))
                self._update_display()

    def _update_display(self) -> None:
        """Update display with current points and instructions"""
        self.display_image = self.image.copy()

        # Draw instructions
        instructions = [
            "Click to place points for parallel lines",
            "Press 'r' to reset current set",
            "Press 'q' to quit",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display_image,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        for i, line_set in enumerate(self.line_sets):
            self._draw_line_set(line_set, self.colors[i % len(self.colors)])

        self._draw_line_set(
            self.current_points, self.colors[len(self.line_sets) % len(self.colors)]
        )

        # Show current state
        status = (
            f"Line set {len(self.line_sets) + 1}: {len(self.current_points)}/4 points"
        )
        cv2.putText(
            self.display_image,
            status,
            (10, self.height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(self.window_name, self.display_image)

    def _draw_line_set(self, line_set: list, color: tuple) -> None:
        # Draw lines
        if len(line_set) >= 2:
            cv2.line(self.display_image, line_set[0], line_set[1], color, 2)
        if len(line_set) >= 4:
            cv2.line(self.display_image, line_set[2], line_set[3], color, 2)

        # Draw points
        for point in line_set:
            cv2.circle(self.display_image, point, 5, color, -1)

    def collect_points(self) -> None:
        print("Collecting points for vanishing point calculation...")

        while True:
            self._update_display()
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                exit()
            elif key == ord("r"):
                self.current_points = []
            elif key == ord("c"):
                if len(self.current_points) == 4:
                    self.line_sets.append(self.current_points)
                    self.current_points = []

                if len(self.line_sets) < 3:
                    print(
                        f"Need {self.n_line_sets} line sets, currently have {len(self.line_sets)}"
                    )

            if len(self.line_sets) == self.n_line_sets:
                break

            if len(self.current_points) == 4:
                self.line_sets.append(self.current_points)
                self.current_points = []

        cv2.destroyWindow(self.window_name)


def show_calibration_guidance() -> None:
    """Display guidance for vanishing point calibration"""
    print("\n=== VANISHING POINT CALIBRATION GUIDANCE ===")
    print(
        "You'll need to identify 3 sets of parallel lines that are perpendicular to each other in the real world."
    )
    print("\nTIPS FOR ACCURATE CALIBRATION:")
    print(
        "1. Choose lines from truly orthogonal structures (walls/floor/ceiling intersections)"
    )
    print("2. Place points for each line as FAR APART as possible")
    print("3. Use high-contrast edges for better precision")
    print("4. Use structures that span a large portion of the image")
    print("\nFor each direction:")
    print("- First line: click 2 points along one edge")
    print("- Second line: click 2 points along another parallel edge")
    print("\nPress Enter to begin...")
    input()


def capture_image(file_dir: str) -> str:
    """Capture image from camera for intrinsic calibration"""
    cap = cv2.VideoCapture(0)
    assert cap is not None, "Error: Could not open camera."

    window_name = "Camera Calibration"
    cv2.namedWindow(window_name)

    print("Camera preview opened. Press SPACE to capture or ESC to cancel.")
    print("Position your camera to capture a scene with clear orthogonal structures.")
    print("Building interiors, boxes, or furniture work well.")

    captured_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            time.sleep(0.5)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("Capture cancelled.")
            break
        elif key == 32:  # SPACE key
            captured_frame = frame.copy()
            break

    cap.release()
    cv2.destroyWindow(window_name)

    assert captured_frame is not None, "Error: No frame captured."

    filename = "calibration_img.jpg"
    filepath = os.path.join(file_dir, filename)
    cv2.imwrite(filepath, captured_frame)
    print(f"Image saved as {filepath}")

    return filename


def calibrate_intrinsics() -> np.ndarray:
    """Calibrate camera intrinsic matrix using vanishing points"""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(file_dir)
    match = re.compile(r"calibration_img\.\w+")

    # Look for existing calibration image in calibration directory
    calibration_img = next((file for file in files if match.match(file)), None)

    if calibration_img is None:
        response = input(
            "No calibration image found. Would you like to take a picture now? (y/n)"
        )
        if response.strip().lower() == "y":
            calibration_img = capture_image(file_dir)
        else:
            print("Calibration cancelled.")
            exit()

    img_path = os.path.join(file_dir, calibration_img)
    img = cv2.imread(img_path)
    assert img is not None, f"Error: Could not load image '{img_path}'."

    show_calibration_guidance()
    point_collector = PointCollector(img)
    point_collector.collect_points()

    vanishing_points = []
    for line_set in point_collector.line_sets:
        try:
            vp = compute_vanishing_point(line_set)
            vanishing_points.append(vp)
        except Exception as e:
            print(f"Error computing vanishing point: {e}")

    if len(vanishing_points) < 3:
        print("Could not compute enough vanishing points.")
        return None

    try:
        K = compute_K_from_vanishing_points(vanishing_points[:3])
        print("Estimated Intrinsic Matrix:")
        print(K)
    except Exception as e:
        print(f"Error computing intrinsic matrix: {e}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np_filename = f"intrinsics.npz"
    file_path = os.path.join(file_dir, np_filename)

    np.savez(
        file_path,
        K=K,
        vanishing_points=vanishing_points,
        img_path=img_path,
        timestamp=timestamp,
    )

    print(f"Saved camera intrinsics to: {np_filename}")

    return K


if __name__ == "__main__":
    calibrate_intrinsics()
