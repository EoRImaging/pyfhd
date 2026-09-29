import numpy as np
import pytest

from pyfhd.pyfhd_tools.pyfhd_utils import weight_invert


@pytest.mark.github_actions
@pytest.mark.parametrize(
    ("weights", "expected", "threshold"),
    [
        (
            np.array([0, 0.01, 0.1, 1.0, 2.0, np.inf, np.nan, -0.01, -0.1, -1.0, -2.0]),
            np.array([0, 100.0, 10.0, 1.0, 0.5, 0, 0, -100.0, -10.0, -1.0, -0.5]),
            None,
        ),
        (
            [0, 0.01, 0.1, 1.0, 2.0, np.inf, np.nan, -0.01, -0.1, -1.0, -2.0],
            [0, 100.0, 10.0, 1.0, 0.5, 0, 0, -100.0, -10.0, -1.0, -0.5],
            None,
        ),
        (
            np.array([0, 0.01, 0.1, 1.0, 2.0, np.inf, np.nan, -0.01, -0.1, -1.0, -2.0]),
            np.array([0, 0, 10.0, 1.0, 0.5, 0, 0, 0, 0, 0, 0]),
            0.02,
        ),
        (
            np.array(
                [0, 0.01, 0.1, 1.0, 2.0, np.inf, np.nan, -0.01, -0.1, -1.0, -2.0],
                dtype=complex,
            ),
            np.array(
                [0, 100.0, 10.0, 1.0, 0.5, 0, 0, -100.0, -10.0, -1.0, -0.5],
                dtype=complex,
            ),
            None,
        ),
        (
            np.array(
                [0, 0.01, 0.1, 1.0, 2.0, np.inf, np.nan, -0.01, -0.1, -1.0, -2.0],
                dtype=complex,
            ),
            np.array([0, 0, 10.0, 1.0, 0.5, 0, 0, 0, -10.0, -1.0, -0.5], dtype=complex),
            0.02,
        ),
        (5, 0.2, None),
        ([5], [0.2], None),
    ],
)
def test_weight_invert(weights, expected, threshold):

    np.testing.assert_allclose(weight_invert(weights, threshold=threshold), expected)
