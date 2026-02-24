import warnings

import croissant as cro
import jax

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
        Tgnd : float
            The ground temperature in Kelvin. Only a constant
            temperature is supported for now.

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
                "Providing sky as an alm instead of a Sky object is"
                "deprecated and will be removed in a future version. Please"
                "switch to using a Sky object for the sky model.",
                FutureWarning,
                stacklevel=2,
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



def correct_ground_loss(vis, fgnd, Tgnd):
    """
    This function is deprecated and will be removed in a future
    version. Please use the `correct_ground_loss` function in
    croissant.simulator instead.

    Correct for ground loss in the simulated visibilities. This
    function recovers the true sky temperature if the assumed ground
    fraction and ground temperature are correct. In simulations, these
    can be accessed with the `compute_fgnd` method of the beam and the
    `Tgnd` attribute of the Simulator, respectively. On real data, they
    have to be estimated or measured.

    Parameters
    ----------
    vis : jax.Array
       The simulated visibilities that include ground loss.
    fgnd : jax.Array
       The assumed ground fraction to use for the correction.
    Tgnd : jax.Array
       The assumed ground temperature to use for the correction. Must
       be spatially uniform in this implementation.

    Returns
    -------
    corrected_vis : jax.Array
       The simulated visibilities with the ground loss corrected.

    """
    warnings.warn(
        "The `correct_ground_loss` function in sim.py is deprecated and "
        "will be removed in a future version. Please use the "
        "`correct_ground_loss` function in croissant.simulator instead.",
        FutureWarning,
        stacklevel=2,
    )
    return cro.simulator.correct_ground_loss(vis, fgnd, Tgnd)
