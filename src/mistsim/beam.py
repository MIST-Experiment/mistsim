from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import s2fft


class Beam(eqx.Module):

    data: jax.Array
    freqs: jax.Array  # in MHz
    horizon: jax.Array  # boolean mask for above/below horizon
    beam_az_rot: jax.Array  # in degrees
    beam_tilt: jax.Array  # in degrees
    lmax: jax.Array
    theta: jax.Array
    phi: jax.Array

    def __init__(
        self, data, freqs, horizon=None, beam_az_rot=0.0, beam_tilt=0.0
    ):
        """
        Beam pattern object. Holds the beam pattern in local antenna
        coordinates and associated metadata. It is assumed that the
        beam is defined on a regular grid in theta/phi coordinates,
        specifically 1 degree spacing in both theta and phi, with theta
        ranging from 0 to 180 degrees (inclusive) and phi ranging from 0
        to 360 degrees (exclusive). This is what s2fft calls the "mwss"
        sampling scheme.

        Parameters
        ----------
        data : array_like
            Power beam pattern data, shape (N_freq, N_theta, N_phi),
            where N_theta = 181 and N_phi = 360.
        freqs : array_like
            Frequencies in MHz corresponding to the beam pattern data.
        horizon : array_like
            The horizon mask: a boolean array specified for each
            (theta, phi) direction, with the same shape as the last two
            axes of data. It is an array with True values for
            directions that are above the horizon and False for
            directions that are below the horizon.
            If None, it is assumed that the horizon is at
            theta = 90 degrees.
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

        """
        if not jnp.isclose(beam_tilt, 0.0):
            raise NotImplementedError("Beam tilt is not yet implemented.")

        self.data = jnp.asarray(data)
        # assumed mwss sampling with 1 degree spacing in theta and phi
        self.lmax = 179  # for 1 degree spacing, lmax = N_theta - 2
        self.L = self.lmax + 1  # useful for s2fft
        self.theta = s2fft.sampling.s2_samples.thetas(
            L=self.L, sampling="mwss"
        )
        self.phi = s2fft.sampling.s2_samples.phis_equiang(
            L=self.L, sampling="mwss"
        )
        ntheta = self.theta.size
        nphi = self.phi.size

        self.freqs = jnp.atleast_1d(freqs)
        nfreq = self.freqs.size

        if self.data.shape != (nfreq, ntheta, nphi):
            raise ValueError(
                f"Data shape {self.data.shape} does not match expected shape "
                f"({nfreq}, {ntheta}, {nphi}) based on freqs and sampling "
                "scheme."
            )

        if horizon is None:
            horizon = self.theta <= 90.0
            self.horizon = horizon[None, :]  # add phi axis

        else:
            self.horizon = jnp.asarray(horizon)

        self.beam_az_rot = jnp.asarray(beam_az_rot)
        self.beam_tilt = jnp.asarray(beam_tilt)

    def _compute_norm(self, use_horizon=True):
        """
        Compute the integral of the beam pattern over the sphere,
        optionally including only the part above the horizon.

        Parameters
        ----------
        use_horizon : bool
            Whether to include only the part of the beam above the
            horizon.
            If False, the entire beam pattern is integrated over.

        Returns
        -------
        norm : jax.Array
            Normalization factor for the beam pattern. One number per
            frequency.

        """
        wgts = s2fft.utils.quadrature_jax.quad_weights_mwss(self.L)
        if use_horizon:
            data = self.data * self.horizon[None]
        else:
            data = self.data

        # data has shape (N_freq, N_theta, N_phi), sum over theta/phi
        norm = jnp.sum(data * wgts[None, :, None], axis=(1, 2))
        return norm

    @jax.jit
    def compute_norm(self):
        """
        Compute the normalization factor for the beam pattern. This is
        the integral of the beam pattern over the whole sphere.

        Returns
        -------
        norm : jax.Array
            Normalization factor for the beam pattern. One number per
            frequency.

        """
        return self._compute_norm(use_horizon=False)

    @jax.jit
    def compute_fgnd(self):
        """
        Compute the ground fraction for the beam pattern. This is the
        integral of the beam pattern over the part of the sphere below
        the horizon, divided by the integral over the whole sphere.

        Returns
        -------
        fgnd : jax.Array
            Ground fraction for the beam pattern. One number per frequency.

        """
        norm_total = self._compute_norm(use_horizon=False)
        norm_above_horizon = self._compute_norm(use_horizon=True)
        fgnd = 1.0 - norm_above_horizon / norm_total
        return fgnd

    @jax.jit
    def compute_alm(self):
        """
        Compute the spherical harmonic coefficients of the beam pattern.
        Only the part of the beam above the horizon is included
        in the spherical harmonic transform. We automatically apply the
        rotations to the beam pattern based on the `beam_az_rot` and
        `beam_tilt` angles.

        Returns
        -------
        alm : jax.Array
            Normalized spherical harmonic coefficients of the beam
            pattern.

        """
        data = self.data * self.horizon[None]  # mask out below-horizon part
        beam2alm = partial(
            s2fft.forward_jax,
            L=self.L,
            sampling="mwss",
            reality=True,
        )
        alm = jax.vmap(beam2alm)(data)
        # now rotate by azimuth XXX
        emms = jnp.arange(-self.lmax, self.lmax + 1)
        phase = jnp.exp(-1j * emms * jnp.radians(self.beam_az_rot))
        alm = alm * phase[None, None, :]  # add freq/ell axes
        return alm
