24127462 - Huynh Trung Ngan
24127020 - Nguyen Thanh Dat

Open `debug.py` and set:

```python
DEBUG_ENABLED = True
```

Run the hider normally. Debug mode creates:

- `log.txt` — turn states, candidate moves, decisions, errors, and runtime.
- `topology_map.txt` — topology scores for the current map.

Set `DEBUG_ENABLED = False` before contests to avoid logging overhead.
