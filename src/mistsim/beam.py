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
    ):
        """
        Beam pattern object. Holds the beam pattern in local antenna
        coordinates and associated metadata. The beam must be defined
        on the grid specified by the `sampling` scheme.

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
            The horizon mask: a boolean array specified for each
            (theta, phi) direction (or pixel), with the same shape as
            the last two (one for healpix) axes of data. It is an array
            with True values for directions that are above the horizon
            and False for directions that are below the horizon.
            If None, it is assumed that the horizon is at
            theta = 90 degrees.
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

        Raises
        ------
        FutureWarning
            If `lmax` is not None.

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
        )
