from functools import partial

from astropy.coordinates import EarthLocation
import crossant.jax as crojax
import equinox as eqx
import jax
import jax.numpy as jnp
import s2fft

from . import Beam, utils


class Simulator(eqx.Module):

    beam: Beam
    sky_alm: jax.Array
    times_jd: jax.Array  # times in Julian day
    freqs: jax.Array  # in MHz
    lon: jax.Array  # longitude of in degrees
    lat: jax.Array  # latitude in degrees
    alt: jax.Array  # altitude in meters
    lmax: jax.Array
    eul_topo: jax.Array  # euler angles for topocentric to eq frame
    dl_topo: jax.Array  # dl array for topocentric to eq frame
    T_gnd: jax.Array  # ground temperature in K

    def __init__(
        self,
        beam,
        sky_alm,
        times_jd,
        freqs,
        lon,
        lat,
        alt=0,
        T_gnd=300.0,
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
            The beam model to use for the simulation. Contains
            information about the beam pattern and any associated
            parameters, including the horizon.
        sky_alm : jax.Array
            The spherical harmonics decomposition of the sky to use for
            the simulation. This should be in equatorial coordinates.
            Expected shape is (Nfreqs, lmax+1, 2*lmax+1).
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
        T_gnd : float
            The ground temperature in Kelvin. Only a constant
            temperature is supported for now.

        """
        if not jnp.all(beam.freqs == freqs):
            raise ValueError("Beam and simulation frequencies do not match.")
        self.freqs = freqs
        self.beam = beam
        self.sky_alm = sky_alm
        self.times_jd = times_jd

        beam_lmax = utils.get_lmax(beam.alm)
        sky_lmax = utils.get_lmax(sky_alm)
        if beam_lmax != sky_lmax:
            raise ValueError("Beam and sky alm have different lmax values.")
        self.lmax = jnp.array(beam_lmax)

        self.T_gnd = jnp.array(T_gnd)

        self.lon = jnp.array(lon)
        self.lat = jnp.array(lat)
        self.alt = jnp.array(alt)
        loc = EarthLocation.from_geodetic(lon, lat, height=alt)
        self.eul_topo, self.dl_topo = crojax.rotations.generate_euler_dl(
            self.lmax, loc, "fk5"
        )

    @jax.jit
    def compute_beam_eq(self):
        """
        Compute the beam alm in equatorial coordinates. This uses the
        pre-computed Euler angles and dl array for the topocentric to
        equatorial transformation.

        Returns
        -------
        beam_eq_alm : jax.Array
            The beam alm in equatorial coordinates. Shape is
            (Nfreqs, lmax+1, 2*lmax+1).

        """
        transform = partial(
            s2fft.utils.rotation.rotate_flms,
            L=self.lmax + 1,
            rotation=self.eul_topo,
            dl_array=self.dl_topo,
        )
        return jax.vmap(transform)(self.beam.alm)

    @jax.jit
    def sim(self):
        beam_eq_alm = self.compute_beam_eq()
        phases = crojax.simulator.rot_alm_z(
            self.lmax, self.times, world="earth"
        )
        # this is the sky contribution
        vis_sky = crojax.simulator.convolve(beam_eq_alm, self.sky_alm, phases)
        # add the ground contribution
        vis = self.beam.fsky * vis_sky + self.beam.fgnd * self.T_gnd
        return vis
