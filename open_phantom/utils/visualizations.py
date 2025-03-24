import cv2
import numpy as np


def visualize_mesh(
    frame: np.ndarray, hand_vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:

    mesh_vis = frame.copy()
    height, width = frame.shape[:2]

    # Draw the base vertices of the mesh (original landmarks)
    for i in range(21):
        pt = tuple(map(int, hand_vertices[i][:2]))
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            cv2.circle(
                mesh_vis, pt, 4, (255, 0, 0), -1
            )  # Red circles for base vertices

    # Draw the interpolated vertices
    for i in range(21, len(hand_vertices)):
        pt = tuple(map(int, hand_vertices[i][:2]))
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            cv2.circle(
                mesh_vis, pt, 2, (0, 255, 255), -1
            )  # Yellow circles for interpolated vertices

    # Draw the faces of the mesh
    for face in faces:
        if len(face) == 3:  # Triangle face
            pt1 = tuple(map(int, hand_vertices[face[0]][:2]))
            pt2 = tuple(map(int, hand_vertices[face[1]][:2]))
            pt3 = tuple(map(int, hand_vertices[face[2]][:2]))

            if (
                0 <= pt1[0] < width
                and 0 <= pt1[1] < height
                and 0 <= pt2[0] < width
                and 0 <= pt2[1] < height
                and 0 <= pt3[0] < width
                and 0 <= pt3[1] < height
            ):
                cv2.line(
                    mesh_vis, pt1, pt2, (0, 255, 0), 1
                )  # Green lines for mesh edges
                cv2.line(mesh_vis, pt2, pt3, (0, 255, 0), 1)
                cv2.line(mesh_vis, pt3, pt1, (0, 255, 0), 1)

    # Add explanation text
    cv2.putText(
        mesh_vis,
        "Red: Base vertices",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        mesh_vis,
        "Yellow: Interpolated vertices",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        mesh_vis,
        "Green: Mesh edges",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return mesh_vis


def visualize_registration(
    cloud: np.ndarray, vertices: np.ndarray, icp_transform: np.ndarray
) -> np.ndarray:

    height, width = 480, 640  # Fixed size visualization
    reg_vis = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Define visualization boundaries
    margin = 50
    view_width = width - 2 * margin
    view_height = height - 2 * margin

    # Find 2D min/max values for scaling
    all_points_2d = np.vstack([cloud[:, :2], vertices[:, :2]])
    min_vals = np.min(all_points_2d, axis=0)
    max_vals = np.max(all_points_2d, axis=0)

    # Function to scale points to fit visualization
    def scale_point(point: np.ndarray) -> tuple:
        scaled = (point - min_vals) / (max_vals - min_vals)
        x = int(scaled[0] * view_width) + margin
        y = int(scaled[1] * view_height) + margin
        return (x, y)

    # Draw original point cloud (red)
    for point in cloud[::10]:  # Downsample for visualization
        pt = scale_point(point[:2])
        cv2.circle(reg_vis, pt, 1, (0, 0, 255), -1)  # Red dots

    # Draw original mesh vertices (blue)
    for vertex in vertices:
        pt = scale_point(vertex[:2])
        cv2.circle(reg_vis, pt, 2, (255, 0, 0), -1)  # Blue dots

    # Apply transformation to mesh vertices
    transformed_vertices = []
    for vertex in vertices:
        # Convert to homogeneous coordinates
        v_hom = np.append(vertex, 1.0)
        # Apply transformation
        v_transformed = np.dot(icp_transform, v_hom)
        transformed_vertices.append(v_transformed[:3])

    # Draw transformed mesh vertices (green)
    for vertex in transformed_vertices:
        pt = scale_point(vertex[:2])
        cv2.circle(reg_vis, pt, 2, (0, 255, 0), -1)  # Green dots

    # Add transformation matrix display
    cv2.putText(
        reg_vis,
        "ICP Transformation Matrix:",
        (10, height - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    for i in range(4):
        matrix_text = f"[{icp_transform[i][0]:.2f}, {icp_transform[i][1]:.2f}, {icp_transform[i][2]:.2f}, {icp_transform[i][3]:.2f}]"
        cv2.putText(
            reg_vis,
            matrix_text,
            (10, height - 90 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    # Add legend
    cv2.putText(
        reg_vis,
        "Red: Point Cloud",
        (width - 200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        reg_vis,
        "Blue: Original Mesh",
        (width - 200, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )
    cv2.putText(
        reg_vis,
        "Green: Transformed Mesh",
        (width - 200, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    return reg_vis


def visualize_constraints(
    frame: np.ndarray,
    refined_landmarks: list,
    constrained_landmarks: np.ndarray,
    camera_intrinsics: tuple,
) -> np.ndarray:

    constraints_vis = frame.copy()
    height, width = frame.shape[:2]
    focal_x, focal_y, center_x, center_y = camera_intrinsics

    # Define finger indices
    thumb_indices = [1, 2, 3, 4]
    index_indices = [5, 6, 7, 8]
    highlighted_indices = thumb_indices + index_indices

    # Draw original refined landmarks
    for i, landmark in enumerate(refined_landmarks):
        x, y, z = landmark
        if z > 0:
            u = int(x * focal_x / z + center_x)
            v = int(y * focal_y / z + center_y)

            if 0 <= u < width and 0 <= v < height:
                if i in highlighted_indices:
                    # Draw larger circles for thumb and index fingers (the constrained ones)
                    cv2.circle(
                        constraints_vis, (u, v), 5, (0, 0, 255), -1
                    )  # Red circles
                else:
                    cv2.circle(constraints_vis, (u, v), 3, (0, 0, 255), -1)

    # Draw connections for original landmarks
    for i in range(len(refined_landmarks) - 1):
        if i + 1 in thumb_indices and i in thumb_indices:
            # Draw thumb connections
            start = refined_landmarks[i]
            end = refined_landmarks[i + 1]

            if start[2] > 0 and end[2] > 0:
                start_u = int(start[0] * focal_x / start[2] + center_x)
                start_v = int(start[1] * focal_y / start[2] + center_y)
                end_u = int(end[0] * focal_x / end[2] + center_x)
                end_v = int(end[1] * focal_y / end[2] + center_y)

                if (
                    0 <= start_u < width
                    and 0 <= start_v < height
                    and 0 <= end_u < width
                    and 0 <= end_v < height
                ):
                    cv2.line(
                        constraints_vis,
                        (start_u, start_v),
                        (end_u, end_v),
                        (0, 0, 255),
                        2,
                    )

    # Draw constrained landmarks
    for i, landmark in enumerate(constrained_landmarks):
        x, y, z = landmark
        if z > 0:
            u = int(x * focal_x / z + center_x)
            v = int(y * focal_y / z + center_y)

            if 0 <= u < width and 0 <= v < height:
                if i in highlighted_indices:
                    # Draw larger circles for thumb and index fingers
                    cv2.circle(
                        constraints_vis, (u, v), 5, (0, 255, 0), -1
                    )  # Green circles
                else:
                    cv2.circle(constraints_vis, (u, v), 3, (0, 255, 0), -1)

    # Draw connections for constrained landmarks
    for i in range(len(constrained_landmarks) - 1):
        if i + 1 in thumb_indices and i in thumb_indices:
            # Draw thumb connections
            start = constrained_landmarks[i]
            end = constrained_landmarks[i + 1]

            if start[2] > 0 and end[2] > 0:
                start_u = int(start[0] * focal_x / start[2] + center_x)
                start_v = int(start[1] * focal_y / start[2] + center_y)
                end_u = int(end[0] * focal_x / end[2] + center_x)
                end_v = int(end[1] * focal_y / end[2] + center_y)

                if (
                    0 <= start_u < width
                    and 0 <= start_v < height
                    and 0 <= end_u < width
                    and 0 <= end_v < height
                ):
                    cv2.line(
                        constraints_vis,
                        (start_u, start_v),
                        (end_u, end_v),
                        (0, 255, 0),
                        2,
                    )

    # Add legend
    cv2.putText(
        constraints_vis,
        "Red: Before constraints",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        constraints_vis,
        "Green: After constraints",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return constraints_vis


def visualize_robot_params(
    frame: np.ndarray,
    position: np.ndarray,
    orientation_matrix: np.ndarray,
    gripper_width: float,
    camera_intrinsics: tuple,
) -> np.ndarray:

    robot_vis = frame.copy()
    height, width = frame.shape[:2]
    focal_x, focal_y, center_x, center_y = camera_intrinsics

    # Extract axes from orientation matrix
    x_axis = orientation_matrix[:, 0]  # First column (principal axis)
    y_axis = orientation_matrix[:, 1]  # Second column
    z_axis = orientation_matrix[:, 2]  # Third column (normal)

    # Project position to image coordinates
    x, y, z = position
    if z > 0:
        u = int(x * focal_x / z + center_x)
        v = int(y * focal_y / z + center_y)

        if 0 <= u < width and 0 <= v < height:
            # Draw end effector position
            cv2.circle(robot_vis, (u, v), 10, (255, 0, 0), -1)  # Blue circle

            # Draw X axis (principal axis)
            dx, dy, dz = x_axis
            scale = 50  # Scale for better visualization
            end_u = int((x + dx * scale) * focal_x / (z + dz * scale) + center_x)
            end_v = int((y + dy * scale) * focal_y / (z + dz * scale) + center_y)

            if 0 <= end_u < width and 0 <= end_v < height:
                cv2.line(
                    robot_vis, (u, v), (end_u, end_v), (0, 0, 255), 2
                )  # Red line (X axis)
                cv2.putText(
                    robot_vis,
                    "X",
                    (end_u, end_v),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            # Draw Z axis (normal)
            nx, ny, nz = z_axis
            end_u = int((x + nx * scale) * focal_x / (z + nz * scale) + center_x)
            end_v = int((y + ny * scale) * focal_y / (z + nz * scale) + center_y)

            if 0 <= end_u < width and 0 <= end_v < height:
                cv2.line(
                    robot_vis, (u, v), (end_u, end_v), (0, 255, 0), 2
                )  # Green line (Z axis)
                cv2.putText(
                    robot_vis,
                    "Z",
                    (end_u, end_v),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Draw Y axis
            yx, yy, yz = y_axis
            end_u = int((x + yx * scale) * focal_x / (z + yz * scale) + center_x)
            end_v = int((y + yy * scale) * focal_y / (z + yz * scale) + center_y)

            if 0 <= end_u < width and 0 <= end_v < height:
                cv2.line(
                    robot_vis, (u, v), (end_u, end_v), (255, 0, 0), 2
                )  # Blue line (Y axis)
                cv2.putText(
                    robot_vis,
                    "Y",
                    (end_u, end_v),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            # Visualize gripper width
            gripper_radius = int(gripper_width * 100)  # Scale for better visualization
            cv2.circle(
                robot_vis, (u, v), gripper_radius, (0, 255, 255), 2
            )  # Yellow circle

            # Add parameter values as text
            y_offset = height - 160
            cv2.putText(
                robot_vis,
                f"Position: ({x:.2f}, {y:.2f}, {z:.2f})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            y_offset += 30
            cv2.putText(
                robot_vis,
                f"X axis: ({x_axis[0]:.2f}, {x_axis[1]:.2f}, {x_axis[2]:.2f})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            y_offset += 30
            cv2.putText(
                robot_vis,
                f"Z axis: ({z_axis[0]:.2f}, {z_axis[1]:.2f}, {z_axis[2]:.2f})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            y_offset += 30
            cv2.putText(
                robot_vis,
                f"Gripper Width: {gripper_width:.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    return robot_vis
