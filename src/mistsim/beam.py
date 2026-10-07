import warnings

import croissant as cro


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
        horizon_frame="beam",
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
            partially visible cells; boolean masks are accepted. Which
            grid the array lives on is set by `horizon_frame`. If None,
            the horizon is at theta = 90 degrees with fractional
            boundary cells (croissant's default). For a horizon given
            as a function of azimuth, ``croissant.horizon_weights``
            builds fractional boundary weights on regular grids.
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
        horizon_frame : {"beam", "topocentric"}
            Forwarded to ``croissant.Beam``. The default ``"beam"``
            keeps `horizon` on the beam grid, so it rotates with
            `beam_az_rot`: right for obstructions attached to the
            antenna, or for masks the caller has already
            counter-rotated. Use ``"topocentric"`` for terrain: the
            weights then live on croissant's fixed ground grid, whose
            phi = 0 is East and phi = 90 deg is North, so a horizon
            given in compass azimuth ``A`` goes at ``phi = 90 - A``
            (deg), independent of `beam_az_rot`. croissant
            counter-rotates it into the beam frame by periodic linear
            interpolation in phi, which is exact for rotations by whole
            grid columns and softens edges in between;
            ``horizon_in_beam_frame`` gives the weights applied.

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
        super().__init__(
            data,
            freqs,
            sampling=sampling,
            horizon=horizon,
            beam_rot=beam_rot,
            beam_tilt=beam_tilt,
            horizon_frame=horizon_frame,
        )
