import numpy as np
import pytest
from scipy.signal.windows import general_cosine

from pyfhd.pyfhd_tools.pyfhd_utils import spectral_window

BH_COEFFS = [0.35875, 0.48829, 0.14128, 0.01168]
BH7_COEFFS = [
    0.27105140069342,
    0.43329793923448,
    0.21812299954311,
    0.06592544638803,
    0.01081174209837,
    0.00077658482522,
    0.00001388721735,
]


@pytest.mark.github_actions
@pytest.mark.parametrize("n_samples", [8, 33, 384])
@pytest.mark.parametrize(
    "window_type, coefficients, power",
    [
        ("Hann", [0.5, 0.5], 1),
        ("Hamming", [0.54, 0.46], 1),
        ("Blackman", [0.42, 0.5, 0.08], 1),
        ("Nutall", [0.355768, 0.487396, 0.144232, 0.012604], 1),
        ("Blackman-Nutall", [0.3635819, 0.4891775, 0.1365995, 0.0106411], 1),
        ("Blackman-Harris", BH_COEFFS, 1),
        ("Blackman-Harris^2", BH_COEFFS, 2),
        ("Blackman-Harris 7-term", BH7_COEFFS, 1),
    ],
)
def test_cosine_sum_windows(n_samples, window_type, coefficients, power):
    window = spectral_window(n_samples, window_type=window_type)
    expected = general_cosine(n_samples, coefficients, sym=True) ** power
    assert window.shape == (n_samples,)
    np.testing.assert_allclose(window, expected, rtol=0, atol=1e-12)


@pytest.mark.github_actions
def test_blackman_harris_7term_shape():
    window = spectral_window(101, window_type="Blackman-Harris 7-term")
    # symmetric, peaks at 1 in the middle and goes to ~0 at the edges
    np.testing.assert_allclose(window, window[::-1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(window[50], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(window[[0, -1]], 0.0, rtol=0, atol=1e-7)


@pytest.mark.github_actions
def test_blackman_harris_7term_periodic():
    # an even periodic window is the first n_samples of the n_samples + 1 window
    window = spectral_window(64, window_type="Blackman-Harris 7-term", periodic=True)
    expected = general_cosine(65, BH7_COEFFS, sym=True)[:64]
    np.testing.assert_allclose(window, expected, rtol=0, atol=1e-12)
    assert np.argmax(window) == 32


@pytest.mark.github_actions
def test_unknown_window_type():
    with pytest.raises(ValueError, match="not recognized"):
        spectral_window(16, window_type="Blackman-Harris 9-term")
