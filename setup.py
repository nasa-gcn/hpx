import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "hpx._core",
            define_macros=[
                ("NPY_TARGET_VERSION", "NPY_1_23_API_VERSION"),
                ("NPY_NO_DEPRECATED_API", "NPY_1_23_API_VERSION"),
            ],
            extra_compile_args=["-std=c++14"],
            include_dirs=[np.get_include()],
            sources=["hpx/_core.cpp"],
        )
    ]
)
