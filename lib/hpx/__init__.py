from importlib.metadata import version as _version

from ._core import LinearSphericalInterpolator

__all__ = ("LinearSphericalInterpolator",)
__version__ = _version(__package__)
