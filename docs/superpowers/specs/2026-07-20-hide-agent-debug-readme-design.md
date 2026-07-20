# Hide Agent Debug README Design

Add `submissions/4/debug/README.md` as a short quick-start guide for the hider diagnostics.

The README will explain how to set `DEBUG_ENABLED = True` in `debug.py`, run the hider normally, inspect `log.txt` and `topology_map.txt`, and disable debugging before contests to avoid logging overhead. Contest result files and testing maps are outside its scope.

Success means a reader can enable debugging, identify both generated diagnostic files, and disable debugging without reading the implementation.
