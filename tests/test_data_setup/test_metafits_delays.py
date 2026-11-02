import numpy as np
import pytest

from pyfhd.data_setup.obs import metafits_delays


@pytest.mark.github_actions
def test_metafits_delays_off_zenith():
    # A non-trivial (off zenith) pointing, every feed should get the same delays
    pattern = [15, 10, 5, 0, 16, 11, 6, 1, 17, 12, 7, 2, 18, 13, 8, 3]
    delays = metafits_delays(",".join(str(d) for d in pattern))
    assert delays.shape == (2, 16)
    assert delays.dtype == np.int64
    np.testing.assert_array_equal(delays, [pattern, pattern])


@pytest.mark.github_actions
def test_metafits_delays_n_feeds():
    delays = metafits_delays("0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3", n_feeds=1)
    np.testing.assert_array_equal(delays, [[0, 1, 2, 3] * 4])
