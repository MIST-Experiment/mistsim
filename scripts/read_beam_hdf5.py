"""
Script to read beam data from an HDF5 file and convert to npz format.
"""

import argparse
import h5py
import numpy as np

def read_beam_hdf5(fpath):
    keys = [
        "frequency",
        "azimuth",
        "elevation",
        "beam_gain",
        "impedance",
        "radiation_efficiency",
        "beam_efficiency",
    ]

    with h5py.File(fpath, "r") as hf:
        out = tuple(np.asarray(hf.get(key)) for key in keys)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read beam data from HDF5 and save as NPZ."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input HDF5 file."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output NPZ file."
    )
    parser.add_argument(
        "--beam_number",
        type=int,
        default=1,
        help="Beam number to read (zero-indexed, default: 1).",
    )
    args = parser.parse_args()

    freqs, az, el, g, z, rad_eff, beam_eff = read_beam_hdf5(args.input_file)
    gain = g[args.beam_number]

    # outputs are 1d-vectors vs frequency
    impedance = z[args.beam_number]
    radiation_efficiency = rad_eff[args.beam_number]
    beam_efficiency = beam_eff[args.beam_number]

    # need to extend gain to the full sphere in theta/phi
    phi = np.deg2rad(az)
    theta = np.linspace(0, np.pi, num=181)  # 0 to 180 degrees in radians
    gain_flip = gain[:, ::-1, :]  # el to theta
    ntheta_below = len(theta) - gain_flip.shape[1]
    gain_below = np.zeros((len(freqs), ntheta_below, len(phi)), dtype=gain.dtype)
    gain_full = np.concatenate((gain_flip, gain_below), axis=1)

    np.savez(
        args.output_file,
        freqs=freqs,
        phi=phi,
        theta=theta,
        gain=gain_full,
        impedance=impedance,
        radiation_efficiency=radiation_efficiency,
        beam_efficiency=beam_efficiency,
    )
    print(f"Beam data has been saved to {args.output_file}")
