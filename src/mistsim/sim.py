import warnings

import croissant as cro

from . import sky as skymod


class Simulator(cro.Simulator):
    def __init__(
        self,
        beam,
        sky,
        times_jd,
        freqs,
        lon,
        lat,
        alt=0,
        lmax=None,
        Tgnd=300.0,
    ):
        """
        Configure a simulation. This class holds all the relevant
        parameters for a simulation and provides necessary methods
        for coordinate transforms and spherical harmonics transforms.

        Note that beam and sky models must be consistent in terms of
        frequencies and lmax values.

        Parameters
        ----------
        beam : Beam
            The beam model to use for the simulation.
        sky : Sky or jax.Array
            The sky model to use for the simulation. Preferred a sky
            object, but sky_alm is also accepted for backwards
            compatibility. If sky_alm is provided, it should be the
            spherical harmonics decomposition of the sky in equatorial
            coordinates. Expected shape is (Nfreqs, lmax+1, 2*lmax+1).
        times_jd : jax.Array
            The times in Julian day at which to simulate the
            observations.
        freqs : jax.Array
            The frequencies in MHz for the simulation. Must be
            consistent with the frequencies used for the beam and the
            sky models.
        lon : float
            The longitude of the observer in degrees.
        lat : float
            The latitude of the observer in degrees.
        alt : float
            The altitude of the observer in meters.
        lmax : int or None
            The maximum ell value to use for the simulation. Must be
            smaller than or equal to the lmax values of the beam and
            sky models. If None, the minimum of the beam and sky lmax
            values is used.
        Tgnd : float or array_like
            The ground temperature in Kelvin, uniform over the ground: a
            scalar, or one value per frequency with shape ``(nfreq,)``,
            such as an emissivity times a physical temperature (passed
            to croissant's Simulator, which checks the shape).

        Raises
        ------
        FutureWarning
            If sky is provided as a jax.Array instead of a Sky object.
            This will be removed in a future version, so users are
            encouraged to switch to using a Sky object for the sky
            model.

        """
        if not isinstance(sky, skymod.Sky):
            warnings.warn(
                "Providing sky as an alm instead of a Sky object is "
                "deprecated and will be removed in a future version. Please "
                "switch to using a Sky object for the sky model.",
                FutureWarning,
                stacklevel=4,
            )
            sky_alm = sky
            sky = skymod._SkyAlm(sky_alm, freqs)

        super().__init__(
            beam,
            sky,
            times_jd,
            freqs,
            lon,
            lat,
            alt=alt,
            lmax=lmax,
            world="earth",
            Tgnd=Tgnd,
        )
