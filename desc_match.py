import numpy as np



def find_descriptors_matches(
    scan_descriptors: np.ndarray[np.float64], ref_descriptors: np.ndarray[np.float64]
) -> tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
    """
    Matching strategy that matches each descriptor with its nearest neighbor in the feature space.

    Args:
        scan_descriptors: Descriptors computed on the dataset of interest.
        ref_descriptors: Descriptors computed on the reference dataset.

    Returns:
        Indices of the matches established.
    """
    non_empty_descriptors = np.any(scan_descriptors, axis=1).nonzero()[0]
    non_empty_ref_descriptors = np.any(ref_descriptors, axis=1).nonzero()[0]
    distance_matrix = cdist(
        scan_descriptors[non_empty_descriptors],
        ref_descriptors[non_empty_ref_descriptors],
    )
    indices = distance_matrix.argmin(axis=1)
    return non_empty_descriptors, non_empty_ref_descriptors[indices]
