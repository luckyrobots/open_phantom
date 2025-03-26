import numpy as np


def compute_line_equation(points: list) -> tuple:
    """Compute line equation from two points"""
    pt1, pt2 = points[0], points[1]
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0] + 1e-10)  # Avoid division by zero
    intercept = pt2[1] - slope * pt2[0]

    a, b, c = -slope, 1.0, -intercept

    return a, b, c


def compute_point_of_intersection(line1: tuple, line2: tuple) -> tuple | None:
    """Compute intersection point of two lines"""
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1 * b2 - a2 * b1
    # If lines are parallel, determinant will be zero
    if abs(det) < 1e-10:
        return None

    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    return (x, y)


def compute_vanishing_point(points: list) -> tuple | None:
    """Compute vanishing point from two parallel lines by intersecting their line equations"""
    a1, b1, c1 = compute_line_equation([points[0], points[1]])
    a2, b2, c2 = compute_line_equation([points[2], points[3]])

    return compute_point_of_intersection((a1, b1, c1), (a2, b2, c2))


def compute_K_from_vanishing_points(vanishing_points: list) -> np.ndarray:
    """
    Compute camera intrinsic matrix K from orthogonal vanishing points.

    Mathematical background:
    - Vanishing points from orthogonal 3D directions provide constraints on the camera's
      intrinsic parameters.
    - If v_i and v_j are vanishing points from orthogonal 3D directions, then:
      v_i^T * omega * v_j = 0, where omega = (K*K^T)^(-1)
    - This gives us quadratic constraints on the elements of omega, which we solve
      to recover K.

    Args:
        vanishing_points: List of at least 3 vanishing points corresponding to
                          orthogonal directions in 3D space.

    Returns:
        K: 3x3 camera intrinsic matrix
    """
    # Each pair of orthogonal vanishing points provides one constraint on omega
    A = []
    for i, point_i in enumerate(vanishing_points):
        for j, point_j in enumerate(vanishing_points):
            if i != j and j > i:  # Consider each pair only once
                # Convert to homogeneous coordinates
                p1 = np.array([point_i[0], point_i[1], 1.0])
                p2 = np.array([point_j[0], point_j[1], 1.0])

                # The constraint v_i^T * omega * v_j = 0 expands to:
                # p1[0]*p2[0]*omega[0,0] + p1[0]*p2[1]*omega[0,1] + p1[0]*omega[0,2]*p2[2] +
                # p1[1]*p2[0]*omega[1,0] + p1[1]*p2[1]*omega[1,1] + p1[1]*omega[1,2]*p2[2] +
                # p1[2]*p2[0]*omega[2,0] + p1[2]*p2[1]*omega[2,1] + p1[2]*p2[2]*omega[2,2] = 0

                # With zero-skew assumption (omega[0,1]=omega[1,0]=0) and
                # square pixels (omega[0,0]=omega[1,1]), we get:
                A.append(
                    [
                        p1[0] * p2[0]
                        + p1[1] * p2[1],  # Coefficient for omega[0,0]=omega[1,1]
                        p1[0] * p2[2]
                        + p1[2] * p2[0],  # Coefficient for omega[0,2]=omega[2,0]
                        p1[1] * p2[2]
                        + p1[2] * p2[1],  # Coefficient for omega[1,2]=omega[2,1]
                        p1[2] * p2[2],  # Coefficient for omega[2,2]
                    ]
                )

    A = np.array(A, dtype=np.float32)

    # Solve the homogeneous system A*w = 0 using SVD
    # The solution is the eigenvector corresponding to the smallest eigenvalue
    _, _, vt = np.linalg.svd(A)

    # Extract the coefficients from the last row of Vt (smallest eigenvalue)
    w1, w4, w5, w6 = vt[
        -1
    ]  # These represent omega[0,0], omega[0,2], omega[1,2], omega[2,2]

    # Reconstruct the omega matrix with enforced symmetry and square pixel constraints
    # omega = (K*K^T)^(-1)
    omega = np.array([[w1, 0.0, w4], [0.0, w1, w5], [w4, w5, w6]])

    # K can be obtained by Cholesky factorization of inverse of omega:
    # If omega = (K*K^T)^(-1), then K*K^T = omega^(-1)
    # Cholesky gives us K^T, so we invert and transpose
    try:
        # Get L such that L*L^T = omega, then K^T = L^(-1)
        K_transpose_inv = np.linalg.cholesky(omega)
        K = np.linalg.inv(K_transpose_inv.T)
        K = K / K[2, 2]  # Normalize so K[2,2] = 1
    except np.linalg.LinAlgError:
        # If omega is not positive definite, find closest PD matrix
        # This can happen due to noise in vanishing point estimation
        eigvals, eigvecs = np.linalg.eigh(omega)
        eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive eigenvalues
        omega_pd = (
            eigvecs @ np.diag(eigvals) @ eigvecs.T
        )  # Reconstruct with positive eigenvalues
        K_transpose_inv = np.linalg.cholesky(omega_pd)
        K = np.linalg.inv(K_transpose_inv.T)
        K = K / K[2, 2]  # Normalize so K[2,2] = 1

    return K
