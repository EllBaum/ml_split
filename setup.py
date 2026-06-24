from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Portability note (heterogeneous clusters): the AVX-512 kernels carry their own
# __attribute__((target("avx512f,avx512vl,avx512dq"))) and are reached only via
# runtime __builtin_cpu_supports dispatch, so they compile and stay fast WITHOUT
# a global -mavx512 flag. Passing -mavx512f globally instead lets -O2 auto-
# vectorize *unguarded* code (tree/msa/merge) with AVX-512 — those instructions
# run unconditionally and SIGILL on any node lacking AVX-512. So the global
# target is AVX2: portable to any AVX2 node, AVX-512 still used at runtime where
# present. If a target node predates AVX2 (pre-2013), drop to ["-msse4.2"].
SIMD = ["-mavx2", "-mfma"]
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