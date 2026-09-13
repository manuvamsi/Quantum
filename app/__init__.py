"""ReQAgnIze edge appliance application package."""

import ctypes
import glob
import os
import platform
import sys


def _preload_lightning_libgomp() -> None:
    """aarch64/Jetson only: dlopen pennylane-lightning's bundled libgomp before cv2/Qt.

    libgomp uses initial-exec TLS; if it is first loaded after the kiosk window
    (Qt) exists, it cannot fit in the static TLS block and lightning.qubit raises
    the misleading "Pre-compiled binaries ... are not available" ImportError.
    Loading it first reserves the static TLS space it needs.
    """
    if not (sys.platform == "linux" and platform.machine() == "aarch64"):
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("pennylane_lightning")
        if spec is None or not spec.submodule_search_locations:
            return
        pkg_dir = os.path.dirname(list(spec.submodule_search_locations)[0])
        libs = sorted(glob.glob(os.path.join(pkg_dir, "pennylane_lightning.libs", "libgomp-*.so*")))
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    except Exception:
        pass


_preload_lightning_libgomp()
