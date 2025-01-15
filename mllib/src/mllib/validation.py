import numpy as np
import numpy.typing as npt


def check_consistent_length(*arrays: npt.NDArray[np.generic]) -> None:
    """
    Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    *arrays : ndarrays
        Arrays to check
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )
