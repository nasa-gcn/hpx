import re

from astropy.coordinates import (
    BaseRepresentation,
    PhysicsSphericalRepresentation,
    uniform_spherical_random_surface,
)
from astropy import units as u
from astropy.utils.misc import NumpyRNGContext
import numpy as np
from scipy.special import sph_harm
import pytest

from hpx._core import LinearSphericalInterpolator


@pytest.mark.parametrize(
    ["points", "values", "message"],
    [
        [np.empty(10), np.empty(10), "object of too small depth for desired array"],
        [np.empty((10, 3)), np.empty((10, 5)), "object too deep for desired array"],
        [np.empty((10, 3)), np.empty(9), "points and values must have the same length"],
        [np.empty((10, 2)), np.empty(10), "points must have shape (npoints, 3)"],
        [
            np.full((10, 3), np.nan),
            np.empty(10),
            "all elements of points must be finite",
        ],
        [
            np.full((10, 3), np.inf),
            np.empty(10),
            "all elements of points must be finite",
        ],
    ],
)
def test_invalid(points, values, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        LinearSphericalInterpolator(points, values)


def astropy_sph_harm(l: int, m: int, points: BaseRepresentation):  # noqa: E741
    points = points.represent_as(PhysicsSphericalRepresentation)
    theta = points.theta.to_value(u.rad)
    phi = points.phi.to_value(u.rad)
    # Caution: scipy.special.sph_harm expects the arguments in the order m, l;
    # not the more conventional order of l, m.
    return sph_harm(m, l, theta, phi)


def astropy_to_xyz(points: BaseRepresentation):
    return points.to_cartesian().xyz.value.T


@pytest.mark.parametrize(["l", "m"], [[l, m] for l in range(3) for m in range(l + 1)])  # noqa: E741
def test_smooth_function(l: int, m: int):  # noqa: E741
    npoints = 100_000
    eval_npoints = 20

    with NumpyRNGContext(1234):
        points = uniform_spherical_random_surface(npoints)
        eval_points = uniform_spherical_random_surface(eval_npoints)

    interp = LinearSphericalInterpolator(
        astropy_to_xyz(points), astropy_sph_harm(l, m, points).real
    )
    actual = interp(astropy_to_xyz(eval_points))
    expected = astropy_sph_harm(l, m, eval_points).real

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0006)
