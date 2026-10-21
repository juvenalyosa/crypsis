"""The C++ pocket-grid extension is optional: without a compiler the install still succeeds and Crypsis uses the
numpy version of the same algorithm (same results, about 15x slower)."""
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    ext_modules = []
else:
    ext_modules = [Pybind11Extension("crypsis._cryptic", ["src/crypsis/_cryptic.cpp"], cxx_std=17,
                                     extra_compile_args=["-O3"])]
    for e in ext_modules:
        e.optional = True


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as exc:  # no compiler
            print(f"crypsis: C++ extension not built ({exc}); using numpy", file=sys.stderr)

    def build_extension(self, ext):
        # conda's Python on macOS repeats its -rpath in the link flags; recent macOS refuses to load a library
        # with a duplicate LC_RPATH, so keep each flag once
        for attr in ("linker_so", "linker_so_cxx"):
            linker = getattr(self.compiler, attr, None)
            if linker:
                seen, kept = set(), []
                for flag in linker:
                    if "rpath" in flag and flag in seen:
                        continue
                    seen.add(flag)
                    kept.append(flag)
                setattr(self.compiler, attr, kept)
        try:
            super().build_extension(ext)
        except Exception as exc:
            print(f"crypsis: C++ extension not built ({exc}); using numpy", file=sys.stderr)


setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
