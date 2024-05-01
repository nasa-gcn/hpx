import numpy as np
import numpy.typing as npt

class LinearSphericalInterpolator:
    def __init__(
        self, points: npt.ArrayLike[np.float64], values: npt.ArrayLike[np.float64]
    ): ...
    def __call__(
        self, points: npt.ArrayLike[np.float64]
    ) -> npt.ArrayLike[np.float64]: ...
