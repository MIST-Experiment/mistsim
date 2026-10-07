import warnings

import croissant as cro
import jax.numpy as jnp
from croissant.horizon import _horizon_in_beam_frame

# mistsim's ground grid is the beam grid as it is at beam_az_rot = 0
# (phi = 0 North, phi = 90 deg West, compass azimuth A = -phi). Croissant's
# ground grid has phi = 0 East (A = 90 - phi). The same direction therefore
# sits 90 deg further round on croissant's grid.
_GROUND_GRID_OFFSET_DEG = 90.0


def _to_croissant_ground_grid(horizon, sampling, spatial_shape):
    """Move a mask from mistsim's ground grid onto croissant's.

    Croissant's own periodic shift does the work: it is exact when 90 deg
    is a whole number of grid columns (the 1-deg MWSS grid, and every
    HEALPix ring) and linear interpolation otherwise. Scalars and
    theta-only masks come back unchanged.
    """
    nside = None
    if sampling == "healpix":
        nside = int(round((spatial_shape[0] / 12) ** 0.5))
    return _horizon_in_beam_frame(
        jnp.asarray(horizon),
        "topocentric",
        _GROUND_GRID_OFFSET_DEG,
        sampling,
        spatial_shape,
        nside,
    )


class Beam(cro.Beam):


    def __init__(
        self,
        data,
        freqs,
        sampling="mwss",
        horizon=None,
        beam_az_rot=0.0,
        beam_tilt=0.0,
        lmax=None,
        horizon_frame="topocentric",
    ):
        """
        Beam pattern object. Holds the beam pattern in local antenna
        coordinates and associated metadata. The beam must be defined
        on the grid specified by the `sampling` scheme.

        Theta is colatitude from zenith and phi is right-handed about
        the zenith, from the beam's x axis towards its y axis. The x
        axis points to compass azimuth `beam_az_rot`, so a beam-grid
        direction has compass azimuth ``A = beam_az_rot - degrees(phi)``
        (mod 360): with ``beam_az_rot = 0``, phi = 90 deg is West.

        Note that the `lmax` parameter is no longer used. The `lmax` is
        automatically determined from the shape of the input data and
        the sampling scheme. To change the `lmax` the simulation runs
        at, change `lmax` of the Simulator object.

        Parameters
        ----------
        data : array_like
            Power beam pattern data. First axis is frequency, second
            axis is theta (colatitude), and third axis is phi (longitude).
            If `sampling` is "healpix", the data only has two dimensions:
            frequency and pixel index.
        freqs : array_like
            Frequencies corresponding to the beam pattern data.
        sampling : str
            Sampling scheme of the beam pattern data. Supported schemes
            are determined by s2fft, currently they include
            {"mw", "mwss", "dh", "gl", "healpix"}. The default is
            "mwss", which is a 1 deg equiangular sampling in theta and
            phi and includes the poles.
        horizon : array_like or None
            Visible fractions in [0, 1] for each (theta, phi) direction
            (or pixel), broadcastable to the spatial axes of data. Zero
            blocks a sample, one keeps it, and fractional values weight
            partially visible cells; boolean masks are accepted. With the
            default ``horizon_frame="topocentric"`` the mask is fixed to
            the ground. It is given on the beam grid as it is at
            ``beam_az_rot = 0`` (phi = 0 North, phi = 90 deg West), so a
            direction at compass azimuth ``A`` sits at ``phi = -A``
            (mod 360 deg), and it stays there whatever `beam_az_rot` is.
            If None, the horizon is at theta = 90 degrees with fractional
            boundary cells (croissant's default). For a horizon given as
            a function of azimuth, ``croissant.horizon_weights`` builds
            fractional boundary weights on regular grids.
        beam_az_rot : float
            Azimuthal rotation of the beam in degrees. The rotation is
            defined in the astronomy convention, i.e., the angle
            measured from the local north towards the local east, with
            0 degrees corresponding to the local north and 90 degrees
            corresponding to the local east.
        beam_tilt : float
            The tilt angle of the beam in degrees. The tilt is the
            angle measured from the local zenith towards the antenna
            pointing direction.
        lmax : int or None
            Removed. Will be ignored if provided and raise a
            FutureWarning.
        horizon_frame : {"topocentric", "beam"}
            Which frame `horizon` is fixed to. **Use the default,
            ``"topocentric"``.** A horizon mask describes what blocks the
            sky from where the antenna stands: terrain, buildings, the
            ground itself. All of these are fixed to the ground, so the
            mask must not turn when the beam does. Structures attached
            to the antenna are not a horizon mask: they belong in the
            beam pattern itself, from the EM simulation.

            ``"beam"`` applies the mask on the rotated beam grid, so it
            turns with `beam_az_rot`. It exists only for masks a caller
            has already counter-rotated into the beam frame by hand (the
            only correct way to handle terrain before this option
            existed). It is the same as ``"topocentric"`` when
            ``beam_az_rot = 0`` and for theta-only masks.

            mistsim moves a topocentric mask onto croissant's ground grid
            (phi = 0 East) before passing it on; croissant then rotates
            it into the beam frame by periodic linear interpolation in
            phi. Both steps are exact when the shifts are whole grid
            columns (e.g. the 1-deg MWSS grid with whole-degree
            `beam_az_rot`) and soften sharp edges otherwise. The `horizon`
            attribute holds croissant's ground-grid weights;
            ``horizon_in_beam_frame`` gives the weights actually applied.

        Raises
        ------
        FutureWarning
            If `lmax` is not None.
        ValueError
            If `horizon_frame` is not "beam" or "topocentric" (raised
            by croissant).

        """
        if lmax is not None:
            warnings.warn(
                "Lmax is now automatically determined from the data shape "
                "and the sampling scheme and will be ignored if provided. "
                "In the future, this will become an error.",
                FutureWarning,
                stacklevel=4,
            )
        # croissant expects X-axis along East
        beam_rot = beam_az_rot - 90
        if horizon is not None and horizon_frame == "topocentric":
            horizon = _to_croissant_ground_grid(
                horizon, sampling, jnp.shape(data)[1:]
            )
        super().__init__(
            data,
            freqs,
            sampling=sampling,
            horizon=horizon,
            beam_rot=beam_rot,
            beam_tilt=beam_tilt,
            horizon_frame=horizon_frame,
        )
