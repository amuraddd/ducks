def notebook_line_magic():
    """
    Avoid having to restart kernel when working with python scripts
    """
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    print("Line Magic Set")
    
def set_ld_library_path():
    """Set path variable for d4rl
    """
    import os, pathlib, importlib.util, glob, shutil
    # ---- Force OSMesa (CPU) build & point to MuJoCo 2.1 ----
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["MUJOCO_PY_FORCE_CPU"] = "1"
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(pathlib.Path.home()/".mujoco/mujoco210")

    # Start LD_LIBRARY_PATH with MuJoCo binaries
    ld_parts = [str(pathlib.Path.home()/".mujoco/mujoco210/bin")]

    # Locate the vendored libglewosmesa.so and add its directory
    spec = importlib.util.find_spec("mujoco_py")
    if not spec:
        raise RuntimeError("mujoco_py not installed in this kernel's interpreter")

    root = spec.submodule_search_locations[0]
    candidates = (
        glob.glob(root + "/vendor/lib/libglewosmesa.so")
        + glob.glob(root + "/generated/_pyxbld_*/**/libglewosmesa.so", recursive=True)
    )
    if not candidates:
        raise RuntimeError(
            "libglewosmesa.so not found under mujoco_py; "
            "ensure OS deps are installed (libosmesa6-dev patchelf pkg-config), then reinstall mujoco-py."
        )
    for c in candidates:
        ld_parts.append(os.path.dirname(c))

    # Apply LD_LIBRARY_PATH (prepend our paths)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts + [os.environ.get("LD_LIBRARY_PATH", "")])

    print("MUJOCO_GL =", os.environ["MUJOCO_GL"])
    print("MUJOCO_PY_MUJOCO_PATH =", os.environ["MUJOCO_PY_MUJOCO_PATH"])
    print("LD_LIBRARY_PATH entries:")
    for p in ld_parts: print("  ", p)