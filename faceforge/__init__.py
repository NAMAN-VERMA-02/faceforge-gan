"""FaceForge: a conditional DCGAN that generates 64x64 faces with controllable attributes."""

from faceforge.config import ATTRIBUTES, Config
from faceforge.models import Discriminator, Generator

__all__ = ["ATTRIBUTES", "Config", "Generator", "Discriminator"]
__version__ = "1.0.0"
