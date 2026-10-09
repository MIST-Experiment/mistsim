"""The mistsim/croissant versions compared against Raul's simulator.

Pure data, importable from any environment: the setup script, the
simulation script and the notebook all read it. Each run is one
environment (a detached mistsim worktree with its own locked venv) and
one ``beam_az_rot``. Two runs may share an environment.

Every run gets the same inputs (``prepare_inputs.py``): Haslam at
nside 128, Raul's FEKO beam, a boolean ``theta <= 80 deg`` mask for
Test 0 and Raul's own UTC time grid. Only the code differs, plus the
beam rotation each version's documentation asks for.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results_versions"
INPUTS = RESULTS / "inputs.npz"

# Detached worktrees of the mistsim repo, one per old environment.
WORKTREE_ROOT = Path(
    "/home/christian/Documents/research/MIST/mistsim-versions"
)

TESTS = {
    0: "Test 0: MARS, 10° blockage",
    1: "Test 1: MARS",
    2: "Test 2: North Pole",
}

# Environments, built by setup_envs.sh: a detached worktree of the
# mistsim commit, synced from its own uv.lock. "over_lock" is what is
# installed over the lock (--no-deps), with the reason. worktree None
# means the repo this file is in.
ENVS = {
    "cro500": {
        "mistsim": "ff99a9d",
        "croissant": "5.0.0",
        "over_lock": (
            "jax 0.5.3: the lock is stale against the pyproject's "
            "jax<0.6 and installs 0.9; 0.5.3 is what cro514 locks"
        ),
    },
    "cro514": {
        "mistsim": "daa5e7b",
        "croissant": "5.1.4",
        "over_lock": None,
    },
    "cro521": {
        "mistsim": "daa5e7b",
        "croissant": "5.2.1",
        "over_lock": (
            "croissant 5.2.1: mistsim never locked 5.2.x; it declares "
            "the same dependencies as 5.1.4"
        ),
    },
    "dev2": {
        "mistsim": "ee3aca5",
        "croissant": "v5.3.0.dev2",
        "over_lock": None,
    },
    "dev3": {
        "mistsim": "03bde4c",
        "croissant": "v5.3.0.dev3",
        "over_lock": None,
    },
    "dev4": {
        "mistsim": "bdfabec",
        "croissant": "v5.3.0.dev4",
        "over_lock": None,
        "worktree": None,
    },
}

# Runs, in chronological order. "changes" lists what this version
# changed relative to the one before it, as far as a scalar Earth
# simulation can see it. A run with "variant_of" changes an input, not
# the code: it is kept out of the version-to-version steps.
RUNS = [
    {
        "key": "cro500_asrun",
        "env": "cro500",
        "label": "5.0.0, as run in Feb",
        "date": "2026-02-25",
        "beam_az_rot": 0.0,
        "changes": [
            "First JAX croissant (s2fft); mistsim's first comparison "
            "with Raul ran on it (ff99a9d).",
            "beam_az_rot was then the angle of the beam X-axis from "
            "East, counter-clockwise, so a North-pointing dipole needed "
            "90. The Feb run passed the default 0.",
            "get_rot_mat built the topocentric frame as North-East-Up, "
            "a left-handed frame (det = -1). At MARS that turned the "
            "beam ~90 deg in azimuth, so the default 0 put the dipole "
            "N-S: the right answer for the wrong input.",
        ],
    },
    {
        "key": "cro500",
        "env": "cro500",
        "label": "5.0.0, documented input",
        "variant_of": "cro500_asrun",
        "date": "2026-02-25",
        "beam_az_rot": 90.0,
        "changes": [
            "Same code, with the input 5.0.0's documentation asks for "
            "(X-axis to North = 90). The NEU frame turns it 90 deg the "
            "wrong way: the dipole lies East-West.",
        ],
    },
    {
        "key": "cro514",
        "env": "cro514",
        "label": "5.1.4 (March)",
        "date": "2026-03-25",
        "beam_az_rot": 0.0,
        "changes": [
            "croissant#110 (5.1.2): the topocentric frame is "
            "East-North-Up, a proper rotation.",
            "croissant 5.1.4 renamed beam_az_rot to beam_rot (N->E); "
            "mistsim #8 made beam_az_rot the compass azimuth of the "
            "X-axis (0 = North), converted as beam_rot = az - 90.",
            "croissant 5.1.3: sky SHT defaults to niter = 0 "
            "(S9 measured +0.003 K from niter = 3).",
            "The baseline of the March update.",
        ],
    },
    {
        "key": "cro521",
        "env": "cro521",
        "label": "5.2.1",
        "date": "2026-04-05",
        "beam_az_rot": 0.0,
        "changes": [
            "croissant#120: gimbal lock handled in rotmat_to_eulerZYZ "
            "(beta near 0, i.e. near the poles).",
            "Run in mistsim daa5e7b's env with croissant swapped; "
            "mistsim never locked 5.2.x.",
        ],
    },
    {
        "key": "dev2",
        "env": "dev2",
        "label": "v5.3.0.dev2",
        "date": "2026-09-14",
        "beam_az_rot": 0.0,
        "changes": [
            "croissant#147: the sky turns about the pole of date "
            "(CIRS at times_jd[0]), not the J2000 pole ~9' away; "
            "mistsim #12 puts the sky alm in that frame.",
            "Sidereal day = one turn of the Earth Rotation Angle (1.5''/day).",
            "Exposed a North Pole bug: aberration in astropy's "
            "topocentric matrix, amplified by the Euler split.",
        ],
    },
    {
        "key": "dev3",
        "env": "dev3",
        "label": "v5.3.0.dev3",
        "date": "2026-09-14",
        "beam_az_rot": 0.0,
        "changes": [
            "croissant#152: Euler angles from the nearest rotation. "
            "Fixes the North Pole (133 deg azimuth error there, "
            "0.004-0.04 deg elsewhere).",
        ],
    },
    {
        "key": "dev4",
        "env": "dev4",
        "label": "v5.3.0.dev4",
        "date": "2026-10-07",
        "beam_az_rot": 0.0,
        "changes": [
            "croissant#155: the default horizon (theta = 90 deg) gives "
            "its boundary row half weight. No effect for this beam: the "
            "FEKO gain on that row is 1e-100. Test 0's explicit boolean "
            "mask is unchanged by design.",
            "croissant#156 / mistsim bdfabec: ground-fixed horizon by "
            "default. A no-op here (beam_az_rot = 0, azimuth-symmetric "
            "mask).",
        ],
    },
]


def worktree(env_key):
    """Directory of the mistsim checkout an environment runs from."""
    env = ENVS[env_key]
    if "worktree" in env and env["worktree"] is None:
        return HERE.parents[2]
    return WORKTREE_ROOT / env_key


def python(env_key):
    """The environment's Python interpreter."""
    return worktree(env_key) / ".venv" / "bin" / "python"


def result_path(run_key):
    return RESULTS / f"{run_key}.npz"
