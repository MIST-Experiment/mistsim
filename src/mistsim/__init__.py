__author__ = "Christian Hellum Bye"
__version__ = "0.0.1"

from jax import config

config.update("jax_enable_x64", True)

from .beam import Beam
from .sim import Simulator
