import os
import sys

import numpy as np


# Global variable to control print redirection
REDIRECT_PRINT = False


def no_print_wrapper(func: callable, *args, **kwargs):
    """Wrapper to suppress stdout and stderr during the execution of a function"""
    if not REDIRECT_PRINT:
        # If REDIRECT_PRINT is False, just call the function normally
        return func(*args, **kwargs)

    # Save the original stdout and stderr
    original_stdout = os.dup(sys.stdout.fileno())
    original_stderr = os.dup(sys.stderr.fileno())

    # Open /dev/null to redirect the outputs
    with open(os.devnull, "w") as fnull:
        # Redirect stdout and stderr to /dev/null
        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())

        try:
            # Execute the function with args and kwargs
            result = func(*args, **kwargs)
        finally:
            # Restore stdout and stderr back to their original file descriptors
            os.dup2(original_stdout, sys.stdout.fileno())
            os.dup2(original_stderr, sys.stderr.fileno())

            # Close the duplicated file descriptors
            os.close(original_stdout)
            os.close(original_stderr)

    return result


def get_max_view_positions(atom_positions: np.ndarray) -> np.ndarray:
    """Rotate atom positions so the best-fit plane aligns with the xy-plane (max view along z)."""
    weights = np.ones(len(atom_positions))
    centroid = np.average(atom_positions, axis=0, weights=weights)
    centered = atom_positions - centroid
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[:, 0]

    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    z_axis = np.array([0, 0, 1])

    rotation_axis = np.cross(normal_vector, z_axis)
    sin_angle = np.linalg.norm(rotation_axis)
    cos_angle = np.dot(normal_vector, z_axis)

    if sin_angle < 1e-8:
        if cos_angle > 0:
            rotation_matrix = np.eye(3)
        else:
            orthogonal = np.array([1, 0, 0]) if abs(normal_vector[0]) < 0.9 else np.array([0, 1, 0])
            rotation_axis = np.cross(normal_vector, orthogonal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.pi
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    else:
        rotation_axis = rotation_axis / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)
        K = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    rot_size = 1  # radians
    extra_rotation = np.array(
        [[np.cos(rot_size), 0, np.sin(rot_size)], [0, 1, 0], [-np.sin(rot_size), 0, np.cos(rot_size)]]
    )
    rotation_matrix = rotation_matrix @ extra_rotation

    return atom_positions @ rotation_matrix.T
