from functools import partial
import jax.numpy as jnp
import s2fft


class Beam:

    def __init__(self, data, frequencies, theta, phi):
        """
        Beam pattern object. Holds the beam pattern in local antenna
        coordinates and associated metadata.

        Parameters
        ----------
        data : array_like
            Power beam pattern data, shape (N_freq, N_theta, N_phi).
            If only one frequency is provided, shape could be
            (N_theta, N_phi).
        frequencies : float or array_like
            Frequencies in MHz corresponding to the beam pattern data.
            If only one frequency is provided, this can be a single
            float.
        theta : array_like
            Theta angles in degrees corresponding to the beam data.
            Measured from zenith and ranging from 0 to 180 degrees.
        phi : array_like
            Phi angles in degrees corresponding to the beam data.
            Measured from the x-axis in the xy-plane and ranging from
            0 to 360 degrees.
        beam_az_rot : float
            Angle between the X-axis of the beam (antenna local frame)
            and the local East direction, in degrees. The angle is
            measured counter-clockwise from the local East direction.
            For example, if the X-axis of the beam points towards the
            local North direction, the `beam_az_rot` would be +90 deg.
        beam_tilt : float
            The tilt angle of the beam in degrees. The tilt is the
            angle measured from the local zenith towards the antenna
            pointing direction.
        horizon : jax.Array
            The horizon mask: a boolean array specified on a Healp
            of ??? with True values
            indicating the directions that are above the horizon. If
            None, no horizon mask is applied.

        """
        self.data = jnp.asarray(data)
        self.frequencies = jnp.atleast_1d(frequencies)
        self.theta = jnp.asarray(theta)
        self.phi = jnp.asarray(phi)

    @classmethod
    def read_FEKO(cls, filename):
        """
        Read FEKO beam pattern from a file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the FEKO beam pattern file.

        Returns
        -------
        Beam
            A Beam object containing the data from the FEKO file.

        """
        # read file
        # d = read...
        # return cls(data, ...)
        raise NotImplementedError

    @property
    def alm(self):
        """
        Compute the spherical harmonic coefficients of the beam pattern.

        Returns
        -------
        alm : array_like
            Spherical harmonic coefficients of the beam pattern.

        """
        raise NotImplementedError
        L = 10  # need to get this from the data shape XXX
        beam2alm = partial(
            s2fft.forward_jax,
            L=L,
            spin=0,
            nside=None,
            sampling="dh",
            reality=True,
        )
        # XXX might have to cut data
        return jax.vmap(beam2alm)(self.data)
