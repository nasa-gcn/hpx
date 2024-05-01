import numpy as np
import numpy.typing as npt

class LinearSphericalInterpolator:
    def __init__(self, points: npt.ArrayLike, values: npt.ArrayLike): ...
    def __call__(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
