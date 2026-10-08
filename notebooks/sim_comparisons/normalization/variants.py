"""The comparison choices tested against Raul's simulator.

Pure data, read by ``run.py`` and by
``raul_comparison_normalization.ipynb``. Each mistsim variant changes
one choice relative to the row it builds on ("after"). The code is the
same for all of them (the current mistsim and croissant); only how the
comparison is set up differs.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results_normalization"
CACHE = RESULTS / "cache"  # *.npz, git-ignored
SUMMARY = RESULTS / "summary.json"

NSIDE = 128
LMAX = 100

TESTS = {
    0: "Test 0: MARS, 10° blockage",
    1: "Test 1: MARS",
    2: "Test 2: North Pole",
}

RAUL_FILES = {
    0: "20260215_for_christian/antenna_temperature_20260215_test1.hdf5",
    1: "20260220_for_christian/antenna_temperature_20260220_test1.hdf5",
    2: "20260220_for_christian/antenna_temperature_20260220_test2.hdf5",
}
HASLAM = "20260215_for_christian/haslam408_ds_Remazeilles2014.fits"
FEKO_OUT = "20260215_for_christian/beam_best_fit_meas_39_joint_wide.out"

# times: "2022" places Raul's LSTs on 2022-07-17 with mean sidereal time
# (the old raul_comparison.ipynb); "raul" is his own UTC grid.
# mask (Test 0 only): "le80" theta <= 80 deg (keeps the 80 deg row,
# edge ~80.5 deg); "half" gives that row weight 0.5 (edge ~80 deg);
# "lt80" blocks the row (edge ~79.5 deg).
# norm: "above_horizon" is Raul's estimator; see raul_tools.run_mistsim.
VARIANTS = [
    {
        "key": "base",
        "label": "old setup",
        "after": None,
        "times": "2022",
        "mirror": False,
        "mask": "le80",
        "change": "The old raul_comparison.ipynb: Raul's LSTs on "
        "2022-07-17 with mean sidereal time.",
    },
    {
        "key": "epoch",
        "label": "+ Raul's time grid",
        "after": "base",
        "times": "raul",
        "mirror": False,
        "mask": "le80",
        "change": "Raul's own UTC grid: 2014-01-01 09:29:45 + i x 359 s, "
        "apparent sidereal time.",
    },
    {
        "key": "mirror",
        "label": "+ mirrored beam",
        "after": "epoch",
        "times": "raul",
        "mirror": True,
        "mask": "le80",
        "change": "Beam gain(-phi): Raul maps phi onto azimuth N->E, "
        "mistsim at beam_az_rot = 0 runs N->W.",
    },
    {
        "key": "edge",
        "label": "+ half-weight edge row",
        "after": "mirror",
        "times": "raul",
        "mirror": True,
        "mask": "half",
        "tests": [0],
        "change": "Test 0's theta = 80 deg row at weight 0.5, so the "
        "horizon sits at 80 deg instead of ~80.5 deg.",
    },
    {
        "key": "edge_lt80",
        "label": "edge row blocked",
        "after": "mirror",
        "times": "raul",
        "mirror": True,
        "mask": "lt80",
        "tests": [0],
        "change": "Test 0's theta = 80 deg row blocked (edge ~79.5 deg): "
        "brackets the edge from the other side.",
    },
    {
        "key": "lmax179",
        "label": "lmax 179",
        "after": "edge",
        "times": "raul",
        "mirror": True,
        "mask": "half",
        "lmax": 179,
        "change": "lmax 100 -> 179.",
    },
    {
        "key": "niter3",
        "label": "sky SHT niter 3",
        "after": "edge",
        "times": "raul",
        "mirror": True,
        "mask": "half",
        "niter": 3,
        "change": "HEALPix sky transform iterated 3 times (default 0).",
    },
    {
        "key": "full_sphere",
        "label": "full-sphere norm",
        "after": "epoch",
        "times": "raul",
        "mirror": False,
        "mask": "le80",
        "norm": "full_sphere",
        "tests": [0],
        "change": "Tgnd = 0 and no correction: divide by the whole beam, "
        "blocked beam sees 0 K (the pipeline's simulate_waterfall).",
    },
    {
        "key": "default",
        "label": "Tgnd = 300 K",
        "after": "epoch",
        "times": "raul",
        "mirror": False,
        "mask": "le80",
        "norm": "default",
        "tests": [0],
        "change": "Simulator.sim with mistsim's default Tgnd = 300 K.",
    },
]

# For "edge" and its descendants, Tests 1 and 2 have no mask edge: they
# use the "mirror" run. The best mistsim configuration per test:
BEST = {0: "edge", 1: "mirror", 2: "mirror"}


def tests_of(v):
    return v.get("tests", list(TESTS))


def run_key(v, k):
    """The cached run a variant uses for test k (edge rows fall back)."""
    if k in tests_of(v):
        return v["key"]
    return "mirror" if v["mask"] == "half" else None


def cache_path(key, k):
    return CACHE / f"ms_{key}_test{k}.npz"
