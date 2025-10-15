import importlib_resources
import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from PyFHD.io.pyfhd_io import recarray_to_dict
from scipy.io import readsav


def dipole_mutual_coupling(
    freq_arr: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """
    Calculate the mutual coupling for a dipole antenna at the given frequencies.

    Parameters
    ----------
    freq_arr : NDArray[np.floating]
        Array of frequencies in Hz.

    Returns
    -------
    NDArray[np.complexfloating]
        Array of mutual coupling values for the dipole antenna.
    """
    # Placeholder implementation, replace with actual mutual coupling calculation
    z_matrix_file = importlib_resources.files(
        "PyFHD.resources.instrument_config"
    ).joinpath("mwa_ZMatrix.fits")
    z_lna_file = importlib_resources.files(
        "PyFHD.resources.instrument_config"
    ).joinpath("mwa_LNA_impedance.sav")
    n_dipole = 16
    n_ant_pol = 2

    # Read the Z matrix and LNA impedance from the files
    z_matrix = fits.open(z_matrix_file)
    z_mat_arr = np.zeros(
        (len(z_matrix), n_ant_pol, n_dipole, n_dipole), dtype=np.complex128
    )
    freq_arr_z_mat = np.zeros(len(z_matrix), dtype=np.float64)
    for ext_i in range(len(z_matrix)):
        z_mat = z_matrix[ext_i].data
        z_mat = z_mat[0] * (np.cos(z_mat[1]) + 1j * np.sin(z_mat[1]))
        freq_arr_z_mat[ext_i] = z_matrix[ext_i].header["FREQ"]
        z_mat_arr[ext_i, 0, :, :] = z_mat[n_dipole:, n_dipole:]
        z_mat_arr[ext_i, 1, :, :] = z_mat[:n_dipole, :n_dipole]
    z_matrix.close()

    z_lna_dict = recarray_to_dict(readsav(z_lna_file)["lna_impedance"])
    z_mat_return = np.zeros(
        (n_ant_pol, freq_arr.size, n_dipole, n_dipole), dtype=np.complex128
    )
    z_mat_interp = np.zeros(
        (freq_arr.size, n_ant_pol, n_dipole, n_dipole), dtype=np.complex128
    )
    for pol_i in range(n_ant_pol):
        for di1 in range(n_dipole):
            for di2 in range(n_dipole):
                z_mat_interp[:, pol_i, di1, di2] = np.interp(
                    freq_arr, freq_arr_z_mat, z_mat_arr[:, pol_i, di1, di2]
                )

    zlna_real = np.interp(freq_arr, z_lna_dict["frequency"], z_lna_dict["z"].real)
    zlna_imag = np.interp(freq_arr, z_lna_dict["frequency"], z_lna_dict["z"].imag)
    zlna_arr = zlna_real + 1j * zlna_imag

    for fi in range(freq_arr.size):
        z_lna = zlna_arr[fi] * np.identity(n_dipole)
        z_inv_x = np.linalg.inv(z_lna + z_mat_interp[fi, 0])
        z_inv_y = np.linalg.inv(z_lna + z_mat_interp[fi, 1])

        # normalize to a zenith pointing, where voltage=Exp(icomp*2.*!Pi*Delay*frequency) and delay=0 so voltage=1.

        norm_test_x = n_dipole / np.abs(np.sum(z_inv_x))
        norm_test_y = n_dipole / np.abs(np.sum(z_inv_y))
        z_inv_x *= norm_test_x
        z_inv_y *= norm_test_y

        z_mat_return[0, fi] = z_inv_x
        z_mat_return[1, fi] = z_inv_y

    return z_mat_return
