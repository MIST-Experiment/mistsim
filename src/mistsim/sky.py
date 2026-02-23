import croissant as cro
import equinox as eqx
import jax


class Sky(cro.Sky):

    def __init__(self, data, freqs, sampling="healpix", coord="galactic"):
        """
        Object that holds the sky model.

        Parameters
        ----------
        data : array_like
            The sky model data. Should be of shape (N_freqs, N_pix) if
            sampling is "healpix" and (N_freqs, N_theta, N_phi) if
            sampling is something else.
        freqs : array_like
            The frequencies corresponding to the sky model data. Should
            have shape (N_freqs,).
        sampling : str
            The sampling scheme of the sky model data. Supported
            schemes are determined by s2fft and include
            {"mw", "mwss", "dh", "gl", "healpix"}. Default is
            "healpix".
        coord : {"galactic", "equatorial"}
            The coordinate system of the sky model data. The alm's will
            be computed in equatorial coordinates.

        Raises
        ------
        ValueError
            If ``coord`` is not one of {"galactic", "equatorial"}.

        """
        if coord not in {"galactic", "equatorial"}:
            raise ValueError(
                f"Unsupported coordinate system: {coord}. Supported systems "
                "are {'galactic', 'equatorial'}."
            )
        super().__init__(data, freqs, sampling=sampling, coord=coord)

    @jax.jit
    def compute_alm_eq(self):
        """
        Compute the spherical harmonic coefficients (alm) of the sky
        model in equatorial coordinates.

        """
        return super().compute_alm(world="earth")


class _SkyAlm:
    _sky_alm: jax.Array
    freqs: jax.Array
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)

    def __init__(self, sky_alm, freqs):
        """
        Create a Sky object from spherical harmonic coefficients (alm),
        defined in equatorial coordinates. This is a class that exists
        to provide backwards compatibility in the Simulator class that
        formerly accepted alm directly. The class is not part of the
        public API and should not be used directly by users.

        Parameters
        ----------
        sky_alm : array_like
            The spherical harmonic coefficients of the sky model in
            equatorial coordinates. Should be of shape
            (N_freqs, lmax+1, 2*lmax+1) where lmax is the harmonic
            bandlimit.

        freqs : array_like
            The frequencies corresponding to the sky model data. Should
            have shape (N_freqs,).

        Returns
        -------
        Sky
            A Sky object with the given alm and frequencies.

        """
        self._sky_alm = jax.asarray(sky_alm)
        self.freqs = freqs
        self.lmax = cro.utils.lmax_from_shape(self._sky_alm.shape)
        self._L = self.lmax + 1

    @jax.jit
    def compute_alm(self):
        return self._sky_alm

    @jax.jit
    def compute_alm_eq(self):
        return self._sky_alm
