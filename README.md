# "I'm Not a Robot" Solver

This script solves the 47th level of https://neal.fun/not-a-robot/

This tool captures a configurable rectangle of the screen, tracks average RGB
changes for the full area and each cell of an evenly divided grid, and
optionally shows a live OpenCV preview while sending debounced keyboard input
(arrow keys by default) whenever a region exceeds a tolerance threshold; run it
via `python screen_monitor.py --left <x> --top <y> --width <w> --height <h>
[options]` and use `--help` to review CLI flags for tuning rows/cols, tolerance,
debounce, FPS, and preview scaling.

## Usage
```
python din_don_dan.py \
    --preview \
    --left=70 \
    --top=700 \
    --width=440 \
    --height=42 \
    --cols=4
```
