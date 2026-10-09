
# Pre-Laboratory 2: Pinch-to-Grab DragRect Engine

A webcam-driven interaction engine built with OpenCV and CVZone. Move your index fingertip over a rectangle, pinch thumb and index together to grab it, and drag it around the screen. It extends the instructor-provided DragRect starter code with a state machine, motion measurement, collision analysis, snap-to-grid and dwell-to-lock.

## Features

- Thumb (landmark 4) to index (landmark 8) pinch with separate grab/release thresholds (hysteresis)
- State machine: `IDLE -> HOVER -> GRABBED -> DRAGGING -> RELEASED`, plus `COLLISION` and `INVALID`
- One object grabbed at a time; overlaps resolve to the nearest centre, then the topmost
- Rectangle bounds, hit-testing and cursor-to-centre distance; objects stay inside the workspace
- Displacement, velocity, speed and acceleration with smoothing and a zero-`dt` guard
- Overlap width, height, area and percentage; invalid drops revert to the start position
- Snap-to-grid (configurable cell size and tolerance) and dwell-to-lock with a visible timer
- Live debug panel and state-dependent colours with a selection outline
- Handles no camera, no hand, incomplete landmarks, off-screen cursor, duplicate key presses and large frame gaps

## Requirements

- Python 3.9 - 3.12 (limited by `mediapipe`)
- A webcam

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Linux, use `opencv-python`, not `opencv-python-headless` (the headless build has no window support).
On macOS, allow camera access for your terminal or IDE when prompted.

## Run

```bash
python prelab/prelab.py
```

Run it from your project folder, because the log files are written to the current directory.

## Controls

| Key | Action |
|-----|--------|
| `c` | Create a new rectangle (placed in a free spot near the cursor) |
| `g` | Toggle snap-to-grid |
| `u` | Unlock all locked rectangles |
| `p` | Write a full-precision snapshot row to `lab2_snapshots.csv` |
| `s` | Save a screenshot |
| `q` | Quit and release the camera |

## Gestures

1. Hover: place your index fingertip inside a rectangle. Hovering alone never moves it.
2. Grab: pinch thumb and index together (below the grab threshold).
3. Drag: move your hand while pinching.
4. Release: open the pinch (above the release threshold).
5. Lock: hold still while grabbed until the timer ring completes. Press `u` to unlock.
6. Unlock: One can also unlock by hovering the index on a locked shape.

## Configuration

Edit the constants at the top of `prelab2_dragrect.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `GRAB_T` / `RELEASE_T` | 40 / 70 px | Pinch close / open thresholds |
| `GRID_SIZE` | 50 px | Snap cell size |
| `SNAP_TOL` | 25 px | Maximum snap error that still snaps |
| `DWELL_TIME` | 1.5 s | Hold-still time to lock |
| `DWELL_RADIUS` / `DWELL_SPEED` | 15 px / 120 px/s | Stability limits for dwell |
| `MAX_OVERLAP_PCT` | 0 % | Overlap allowed on drop |
| `LOSS_GRACE` | 0.4 s | How long a lost hand keeps its object |

## Logs for the report

- `lab2_events.csv`: timestamped state changes and errors, useful for test-case timestamps
- `lab2_snapshots.csv`: raw and smoothed cursor, motion, pinch, collision and grid values for the worked examples

Both files are appended on each run and ignored by git.

## Math used

- Bounds: `L = cx - w/2`, `R = cx + w/2`, `T = cy - h/2`, `B = cy + h/2`
- Distance: `d = sqrt((x2 - x1)^2 + (y2 - y1)^2)`
- Motion: `dp = p_t - p_(t-1)`, `v = dp/dt`, `speed = sqrt(vx^2 + vy^2)`, `a = (v_t - v_(t-1))/dt`
- Pinch: `dp = sqrt((x8 - x4)^2 + (y8 - y4)^2)`
- Intersection: `wi = max(0, min(R1, R2) - max(L1, L2))`, `Ai = wi * hi`, `overlap% = 100 * Ai / min(A1, A2)`
- Grid snap: `xs = g * round(x/g)`, snap error = distance to `(xs, ys)`
- Dwell: `t_dwell = t_now - t_enter`

## Troubleshooting

- **"No webcam found"**: close other apps using the camera and check OS camera permissions.
- **Slow camera start on Windows**: try `cv2.VideoCapture(idx, cv2.CAP_DSHOW)`.
- **Pinch flickers or never triggers**: adjust `GRAB_T` and `RELEASE_T` for your camera distance.
- **Jittery drag**: lower `EMA_CURSOR` for more smoothing.
