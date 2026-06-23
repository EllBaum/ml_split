from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Same flags as the Makefile: AVX-512 (per-function target attrs + runtime
# dispatch in the kernels), counters/profiler stripped.
SIMD = ["-mavx2", "-mfma", "-mavx512f", "-mavx512cd",
        "-mavx512dq", "-mavx512bw", "-mavx512vl"]
DEFS = ["-DCOUNTERS_DISABLED", "-DPROF_DISABLED"]

ext = Pybind11Extension(
    "ml_split",
    sources=[
        "python/bindings.cpp",
        "src/likelihood_scorer_unit.cpp",   # umbrella TU (#includes the pieces)
        "src/tree.cpp",
        "src/msa.cpp",
        "src/merge_prep.cpp",
        "src/merge_session.cpp",
    ],
    include_dirs=["src"],
    cxx_std=17,
    extra_compile_args=["-O2"] + SIMD + DEFS,
)

setup(
    name="ml_split",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
