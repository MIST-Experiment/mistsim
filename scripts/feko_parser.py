import re
from argparse import ArgumentParser

import numpy as np


def parse_feko_out(filepath, output_filepath=None):
    """
    Parses a Feko .out file to extract Far Field gain data.

    Parameters
    ----------
    filepath : str
        Path to the .out file.
    output_filepath : str
        Path to save the .npz file. If None, saves with same name as input.

    """

    freqs = []
    # This list will hold a list of gain values for each frequency found
    raw_gain_data = []

    # Temporary holders for the current block being processed
    current_freq_val = None
    current_block_data = []
    reading_data = False

    # Regex to find frequency. Feko format usually: "FREQUENCY =   1.2345E+09"
    # We look for "FREQ" and then numbers.
    freq_pattern = re.compile(r"FREQ\s*=\s*([0-9\.\-E\+]+)", re.IGNORECASE)

    print(f"Reading file: {filepath}...")

    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        freq_match = freq_pattern.search(line)
        if freq_match:
            if current_block_data:
                raw_gain_data.append(current_block_data)
                current_block_data = []
                reading_data = False

            # feko frequencies are usually in Hz
            current_freq_val = float(freq_match.group(1))
            freqs.append(current_freq_val)
            continue

        # 2. Detect Table Header
        # User specified headers: THETA PHI magn. phase ...
        if (
            line.startswith("THETA")
            and "PHI" in line
            and "total" in line.lower()
        ):
            reading_data = True
            continue

        # 3. Read Data Rows
        if reading_data:
            if not line:
                # Empty line usually signifies end of the table
                reading_data = False
                if current_block_data:
                    raw_gain_data.append(current_block_data)
                    current_block_data = []
                continue

            parts = line.split()

            # Check if line is numeric (start of data)
            # Feko sometimes puts '.....' lines or text under headers
            try:
                # Attempt to parse the first item (Theta)
                theta = float(parts[0])
                phi = float(parts[1])

                # Column 8 (0-indexed) is specified as 'total'
                # Cols: 0=Theta, 1=Phi, 8=Total gain in dB
                total_gain = float(parts[8])

                current_block_data.append((theta, phi, total_gain))

            except (ValueError, IndexError):
                # if we fail to parse numbers, we assume we hit the end
                # of the table text or a separator line
                if len(current_block_data) > 0:
                    reading_data = False
                    raw_gain_data.append(current_block_data)
                    current_block_data = []

    # Handle case where file ends while reading data
    if current_block_data:
        raw_gain_data.append(current_block_data)

    # ---------------------------------------------------------
    # Processing and Reshaping
    # ---------------------------------------------------------

    if not raw_gain_data:
        print("Error: No data found. Check file format.")
        return

    # Convert to numpy arrays for easier handling
    # We use the first frequency block to determine Grid shape (Ntheta, Nphi)
    ref_block = np.array(raw_gain_data[0])

    # Extract unique sorted Thetas and Phis to determine grid axes
    unique_thetas = np.unique(ref_block[:, 0])
    unique_phis = np.unique(ref_block[:, 1])

    Ntheta = len(unique_thetas)
    Nphi = len(unique_phis)
    Nfreqs = len(freqs)

    print(f"Found {Nfreqs} Frequencies.")
    print(f"Grid detected: Ntheta={Ntheta}, Nphi={Nphi}")

    # Initialize the final 3D array (F x T x P)
    gain_matrix = np.zeros((Nfreqs, Ntheta, Nphi))

    # We need to map the flat lists to the 3D grid.
    # Feko output order is deterministic, but mapping by value is safer
    # in case sorting differs.

    # Create index maps for speed
    theta_map = {val: i for i, val in enumerate(unique_thetas)}
    phi_map = {val: i for i, val in enumerate(unique_phis)}

    for f_idx, block in enumerate(raw_gain_data):
        block = np.array(block)
        # Check consistency
        if len(block) != Ntheta * Nphi:
            print(
                f"Warning: Frequency {freqs[f_idx]} has {len(block)} points,"
                f"expected {Ntheta*Nphi}. Skipping."
            )
            continue

        for row in block:
            t_val = row[0]
            p_val = row[1]
            g_val = row[2]

            t_idx = theta_map.get(t_val)
            p_idx = phi_map.get(p_val)

            if t_idx is not None and p_idx is not None:
                gain_matrix[f_idx, t_idx, p_idx] = g_val

    # convert gain from dB to linear scale
    gain_matrix = 10 ** (gain_matrix / 10)

    # ---------------------------------------------------------
    # Save to .npz
    # ---------------------------------------------------------
    if output_filepath is None:
        output_filepath = filepath.replace(".out", ".npz")
        if output_filepath == filepath:
            output_filepath += ".npz"

    np.savez(
        output_filepath,
        freqs=np.array(freqs),
        theta=unique_thetas,
        phi=unique_phis,
        gain=gain_matrix,
    )

    print(f"Successfully converted. Saved to: {output_filepath}")
    print(f"Gain shape: {gain_matrix.shape}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Parse Feko .out files to extract Far Field gain data."
    )
    parser.add_argument("filepath", help="Path to the Feko .out file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the output .npz file. Default: same name as input.",
    )
    args = parser.parse_args()
    filepath = args.filepath
    output_filepath = args.output
    parse_feko_out(filepath, output_filepath=output_filepath)
