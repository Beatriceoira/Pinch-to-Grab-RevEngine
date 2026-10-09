#!/usr/bin/env python3
"""
PRE-LABORATORY 2 - Pinch-to-grab DragRect engine (OpenCV + CVZone)

Keys:  c = new rectangle   g = grid on/off   u = unlock all
       p = write snapshot row to lab2_snapshots.csv (raw values for worked examples)
       s = save screenshot  q = quit
Log files (appended, timestamps relative to program start):
       lab2_events.csv     state changes / errors  -> use for test-case timestamps
       lab2_snapshots.csv  full-precision values   -> use for manual verification
"""
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

# ----------------------------------------------------------------- CONFIG
FRAME_W, FRAME_H = 1280, 720
GRAB_T = 40.0          # px: pinch closes below this  -> grab
RELEASE_T = 70.0       # px: pinch opens above this   -> release (hysteresis)
EMA_CURSOR = 0.6       # weight of new sample in cursor smoothing (0..1)
EMA_V = 0.5            # weight of new sample in velocity smoothing
MAX_DT = 0.25          # s: larger frame gaps restart motion estimation
MIN_DT = 1e-3          # s: guard against divide-by-zero
LOSS_GRACE = 0.4       # s: hand may vanish this long while holding an object
DRAG_DEADZONE = 6.0    # px: GRABBED -> DRAGGING once cursor moves this far
GRID_SIZE = 50         # px: snap-to-grid cell size (g)
SNAP_TOL = 25.0        # px: snap only if snap error <= this
DWELL_TIME = 1.5       # s: hold still this long while grabbed -> lock
DWELL_RADIUS = 15.0    # px: cursor must stay within this of dwell anchor
DWELL_SPEED = 120.0    # px/s: must also be slower than this
MAX_OVERLAP_PCT = 0.0  # % overlap allowed on drop (0 = no overlap allowed)
FLASH_TIME = 0.4       # s: how long RELEASED / INVALID is shown
MAX_READ_FAILS = 50    # consecutive failed frame reads before clean exit

IDLE, HOVER, GRABBED, DRAGGING = "IDLE", "HOVER", "GRABBED", "DRAGGING"
RELEASED, COLLISION, INVALID = "RELEASED", "COLLISION", "INVALID"
STATE_COLORS = {            # BGR
    IDLE: (255, 0, 255),
    HOVER: (0, 255, 255),
    GRABBED: (0, 165, 255),
    DRAGGING: (0, 255, 0),
    RELEASED: (255, 255, 0),
    COLLISION: (0, 0, 255),
    INVALID: (0, 0, 255),
}
LOCKED_COLOR = (200, 120, 0)


# ----------------------------------------------------------------- MATH
def dist(a, b):
    """d = sqrt((x2-x1)^2 + (y2-y1)^2)"""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def snap_point(x, y, g):
    """xs = g*round(x/g); snap error = distance to the snapped point."""
    xs = g * math.floor(x / g + 0.5)
    ys = g * math.floor(y / g + 0.5)
    return xs, ys, math.sqrt((x - xs) ** 2 + (y - ys) ** 2)


class DragRect:
    _count = 0

    def __init__(self, pos_center, size=(200, 200)):
        DragRect._count += 1
        self.id = DragRect._count
        self.posCenter = [float(pos_center[0]), float(pos_center[1])]
        self.size = [float(size[0]), float(size[1])]
        self.locked = False

    @property
    def bounds(self):
        """L = cx-w/2, R = cx+w/2, T = cy-h/2, B = cy+h/2"""
        cx, cy = self.posCenter
        w, h = self.size
        return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2

    @property
    def area(self):
        return self.size[0] * self.size[1]

    def contains(self, p):
        L, R, T, B = self.bounds
        return L < p[0] < R and T < p[1] < B

    def distance_to(self, p):
        return dist(p, self.posCenter)

    def clamp(self, W, H):
        """Keep the whole rectangle inside the visible workspace."""
        w, h = self.size
        self.posCenter[0] = min(max(self.posCenter[0], w / 2), W - w / 2)
        self.posCenter[1] = min(max(self.posCenter[1], h / 2), H - h / 2)


def intersection(a, b):
    """wi, hi, Ai and overlap% relative to the smaller rectangle."""
    L1, R1, T1, B1 = a.bounds
    L2, R2, T2, B2 = b.bounds
    wi = max(0.0, min(R1, R2) - max(L1, L2))
    hi = max(0.0, min(B1, B2) - max(T1, T2))
    Ai = wi * hi
    smaller = min(a.area, b.area)
    pct = 100.0 * Ai / smaller if smaller > 0 else 0.0
    return wi, hi, Ai, pct


def worst_collision(rect, rects):
    """Returns (other, wi, hi, Ai, pct) for the largest overlap, or None."""
    best = None
    for o in rects:
        if o is rect:
            continue
        wi, hi, Ai, pct = intersection(rect, o)
        if Ai > 0 and (best is None or Ai > best[3]):
            best = (o, wi, hi, Ai, pct)
    return best


def pick_target(rects, p):
    """Nearest centre among rectangles under the cursor; ties -> topmost."""
    best, bd = None, None
    for r in rects:                      # later in list = drawn on top
        if r.contains(p):
            d = r.distance_to(p)
            if best is None or d <= bd:
                best, bd = r, d
    return best


def find_free_spot(target, size, rects, W, H, step=25):
    probe = DragRect(target, size)
    best, bd = None, None
    w, h = size
    x = w / 2
    while x <= W - w / 2:
        y = h / 2
        while y <= H - h / 2:
            probe.posCenter = [x, y]
            if worst_collision(probe, rects) is None:
                d = dist((x, y), target)
                if best is None or d < bd:
                    best, bd = (x, y), d
            y += step
        x += step
    return best


def resolve_drop(rect, start_center, rects, grid_on, W, H):
    """Snap (if enabled and error within tolerance), validate overlap,
    revert to the pre-grab position if the drop is invalid."""
    snapped, err = False, 0.0
    if grid_on:
        xs, ys, err = snap_point(rect.posCenter[0], rect.posCenter[1], GRID_SIZE)
        if err <= SNAP_TOL:
            rect.posCenter = [xs, ys]
            rect.clamp(W, H)
            snapped = True
    col = worst_collision(rect, rects)
    if col is not None and col[4] > MAX_OVERLAP_PCT:
        rect.posCenter = list(start_center)
        return False, snapped, err
    return True, snapped, err

class Motion:
    """Smoothed cursor, displacement, velocity, speed, acceleration."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.p = None
        self.t = None
        self.dx = self.dy = self.dt = 0.0
        self.vx = self.vy = self.speed = 0.0
        self.ax = self.ay = self.acc = 0.0
        self.have_v = False
        self.gap = True

    def update(self, raw, t):
        if self.p is None:
            self.p, self.t, self.gap = (raw[0], raw[1]), t, True
            return self.p
        dt = t - self.t
        if dt > MAX_DT:                  # large frame-time gap: restart
            self.reset()
            self.p, self.t = (raw[0], raw[1]), t
            return self.p
        dt = max(dt, MIN_DT)             # never divide by zero
        self.gap = False
        px, py = self.p
        sx = px + EMA_CURSOR * (raw[0] - px)
        sy = py + EMA_CURSOR * (raw[1] - py)
        self.dx, self.dy, self.dt = sx - px, sy - py, dt          # dp = p_t - p_t-1
        vx, vy = self.dx / dt, self.dy / dt                        # v = dp/dt
        if self.have_v:
            vx = self.vx + EMA_V * (vx - self.vx)
            vy = self.vy + EMA_V * (vy - self.vy)
            self.ax = (vx - self.vx) / dt                          # a = dv/dt
            self.ay = (vy - self.vy) / dt
        else:
            self.ax = self.ay = 0.0
        self.acc = math.sqrt(self.ax ** 2 + self.ay ** 2)
        self.vx, self.vy = vx, vy
        self.speed = math.sqrt(vx ** 2 + vy ** 2)
        self.have_v = True
        self.p, self.t = (sx, sy), t
        return self.p

class Logger:
    def __init__(self, t0):
        self.t0 = t0
        self.ef, self.ew = self._open("lab2_events.csv", ["time_s", "event"])
        self.sf, self.sw, self.s_header = None, None, False

    @staticmethod
    def _open(path, header):
        new = not os.path.exists(path)
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(header)
        return f, w

    def event(self, text):
        t = time.time() - self.t0
        print(f"[{t:8.3f}s] {text}")
        self.ew.writerow([f"{t:.3f}", text])
        self.ef.flush()

    def snapshot(self, info):
        new = not os.path.exists("lab2_snapshots.csv")
        if self.sf is None:
            self.sf = open("lab2_snapshots.csv", "a", newline="")
            self.sw = csv.writer(self.sf)
        if new:
            self.sw.writerow(list(info.keys()))
        self.sw.writerow([info[k] for k in info])
        self.sf.flush()

    def close(self):
        for f in (self.ef, self.sf):
            if f:
                f.close()

def open_camera():
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Using camera index {idx}")
            cap.set(3, FRAME_W)
            cap.set(4, FRAME_H)
            return cap
        cap.release()
    return None

def f(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def put(img, text, org, scale=0.55, color=(255, 255, 255), th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_panel(out, info, H):
    lines = [
        f"STATE {info['state']}   OBJ #{info['obj']}{' [LOCKED]' if info['locked'] else ''}   hand: {info['hand']}",
        f"Bounds L{f(info['L'], '.0f')} R{f(info['R'], '.0f')} T{f(info['T'], '.0f')} B{f(info['B'], '.0f')}",
        f"Cursor->centre d = {f(info['dist_center'])} px",
        f"Disp dx={f(info['dx'], '.2f')} dy={f(info['dy'], '.2f')} px   dt={f(info['dt'] and info['dt'] * 1000, '.1f')} ms",
        f"Speed {f(info['speed'])} px/s   Acc {f(info['acc'], '.0f')} px/s2",
        f"Pinch dp={f(info['pinch'])} (grab<{GRAB_T:.0f} rel>{RELEASE_T:.0f})  {'PINCHED' if info['pinching'] else 'open'}",
        f"Collision {f(info['col_wi'], '.0f')}x{f(info['col_hi'], '.0f')}={f(info['col_area'], '.0f')} px2 "
        f"({f(info['col_pct'])}%) with #{info['col_other'] if info['col_other'] else '-'}",
        f"Grid {'ON' if info['grid_on'] else 'OFF'} g={GRID_SIZE} cell=({f(info['grid_x'], '.0f')},{f(info['grid_y'], '.0f')}) "
        f"snap err={f(info['snap_err'])} px",
        f"Dwell {f(info['dwell'], '.2f')} / {DWELL_TIME:.1f} s    FPS {info['fps']:.0f}",
        "c new | g grid | u unlock | p snapshot | s shot | q quit",
    ]
    lh = 22
    h = lh * len(lines) + 12
    x0, y0 = 10, H - h - 10
    roi = out[y0:y0 + h, x0:x0 + 600]
    if roi.size:
        out[y0:y0 + h, x0:x0 + 600] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, ln in enumerate(lines):
        put(out, ln, (x0 + 8, y0 + 20 + i * lh), 0.5)

def main():
    cap = open_camera()
    if cap is None:
        print("ERROR: No webcam found. Check that it is connected and not used by another app.")
        return 1

    W, H = FRAME_W, FRAME_H
    for _ in range(10):
        ok, probe = cap.read()
        if ok and probe is not None:
            H, W = probe.shape[:2]
            break

    rs = min(200, W // 6, H // 3)                       
    gap = (W - rs - 100) / 4
    rects = [DragRect([rs / 2 + 50 + i * gap, rs / 2 + 50], (rs, rs)) for i in range(5)]

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    t0 = time.time()
    log = Logger(t0)
    motion = Motion()

    state = IDLE
    selected = None
    flash_obj = None
    state_until = 0.0
    pinching = prev_pinching = False
    offset = (0.0, 0.0)
    grab_point = None
    start_center = None
    lost_since = None
    dwell_anchor, dwell_t0, dwell_t = None, 0.0, 0.0
    grid_on = True
    last_create = 0.0
    fails = 0
    msg, msg_until = "", 0.0
    prev_frame_t, fps = time.time(), 0.0
    now = time.time()

    def release(reason, lock=False):
        nonlocal state, state_until, selected, flash_obj, dwell_anchor
        rect = selected
        valid, snapped, err = resolve_drop(rect, start_center, rects, grid_on, W, H)
        if valid and lock:
            rect.locked = True
        flash_obj = rect
        state = RELEASED if valid else INVALID
        state_until = now + FLASH_TIME
        selected, dwell_anchor = None, None
        extra = f", snapped (err {err:.1f}px)" if snapped else ""
        log.event(f"{reason}: #{rect.id} {'LOCKED' if (lock and valid) else 'released'} "
                  f"{'VALID' + extra if valid else 'INVALID drop -> reverted'} "
                  f"at ({rect.posCenter[0]:.1f},{rect.posCenter[1]:.1f})")

    log.event(f"start: workspace {W}x{H}, {len(rects)} rectangles of {rs}px")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails == 1 or fails % 10 == 0:
                    log.event(f"camera read failed ({fails})")
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or fails >= MAX_READ_FAILS:
                    break
                continue
            fails = 0

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_t, MIN_DT))
            prev_frame_t = now

            img = cv2.flip(frame, 1)
            H, W = img.shape[:2]
            hand_status = "OK"
            try:
                hands, img = detector.findHands(img, draw=False)
            except Exception as e:                         # detector hiccup: skip frame
                hands, hand_status = [], "DETECTOR ERROR"
                log.event(f"detector error: {e}")

            if state in (RELEASED, INVALID) and now >= state_until:
                state, flash_obj = IDLE, None

            raw, pinch_d, thumb = None, None, None
            if hands:
                lm = hands[0].get("lmList", [])
                if len(lm) >= 21:
                    x8, y8 = lm[8][:2]
                    x4, y4 = lm[4][:2]
                    if 0 <= x8 < W and 0 <= y8 < H:
                        raw = (float(x8), float(y8))
                        thumb = (int(x4), int(y4))
                        pinch_d = math.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2)
                    else:
                        hand_status = "CURSOR OFF-SCREEN"
                else:
                    hand_status = "INCOMPLETE LANDMARKS"
            else:
                if hand_status == "OK":
                    hand_status = "NO HAND"

            key = cv2.waitKey(1) & 0xFF
            cursor, hover_obj = None, None
            dwell_active = False

            if raw is None:
                motion.reset()
                if selected is not None:
                    if lost_since is None:
                        lost_since = now
                        log.event(f"hand lost while holding #{selected.id} ({hand_status})")
                    elif now - lost_since > LOSS_GRACE:
                        release("hand loss timeout")
                        pinching = prev_pinching = False
                        lost_since = None
                elif state not in (RELEASED, INVALID):
                    state = IDLE
            else:
                lost_since = None
                cursor = motion.update(raw, now)
                if motion.gap:
                    dwell_anchor = None

                prev_pinching = pinching
                if pinching:
                    if pinch_d > RELEASE_T:
                        pinching = False
                elif pinch_d < GRAB_T:
                    pinching = True
                rising = pinching and not prev_pinching

                hover_obj = pick_target(rects, cursor)

                if selected is None:
                    if not (state in (RELEASED, INVALID) and now < state_until):
                        state = HOVER if hover_obj else IDLE
                        if rising:
                            if hover_obj is None:
                                msg, msg_until = "Pinch ignored: nothing under cursor", now + 1.2
                            elif hover_obj.locked:
                                msg, msg_until = f"#{hover_obj.id} is locked (press u)", now + 1.2
                            else:
                                selected = hover_obj
                                start_center = list(selected.posCenter)
                                offset = (selected.posCenter[0] - cursor[0],
                                          selected.posCenter[1] - cursor[1])
                                grab_point = cursor
                                state = GRABBED
                                dwell_anchor, dwell_t0, dwell_t = cursor, now, 0.0
                                rects.remove(selected)
                                rects.append(selected)       # bring to top
                                log.event(f"grab #{selected.id} at cursor "
                                          f"({cursor[0]:.1f},{cursor[1]:.1f}) pinch={pinch_d:.1f}px")
                else:
                    invalid = False
                    if not pinching:
                        release("pinch released")
                    else:
                        if state == GRABBED and dist(cursor, grab_point) > DRAG_DEADZONE:
                            state = DRAGGING
                            log.event(f"dragging #{selected.id}")
                        if state != GRABBED:
                            selected.posCenter = [cursor[0] + offset[0], cursor[1] + offset[1]]
                            selected.clamp(W, H)
                            col = worst_collision(selected, rects)
                            invalid = col is not None and col[4] > MAX_OVERLAP_PCT
                            new_state = COLLISION if invalid else DRAGGING
                            if new_state != state and new_state == COLLISION:
                                log.event(f"collision flagged: #{selected.id} overlaps #{col[0].id} "
                                          f"({col[4]:.1f}%)")
                            state = new_state
                        moved = (dwell_anchor is None
                                 or dist(cursor, dwell_anchor) > DWELL_RADIUS
                                 or motion.speed > DWELL_SPEED)
                        if moved or invalid:
                            dwell_anchor, dwell_t0 = cursor, now
                        dwell_t = now - dwell_t0
                        dwell_active = True
                        if dwell_t >= DWELL_TIME:
                            release("dwell lock", lock=True)
                            dwell_active = False

            if key == ord('c') and now - last_create > 0.3:        # debounce duplicates
                last_create = now
                target = cursor if cursor else (W / 2, H / 2)
                spot = find_free_spot(target, (rs, rs), rects, W, H)
                if spot:
                    nr = DragRect(spot, (rs, rs))
                    rects.append(nr)
                    log.event(f"created #{nr.id} at ({spot[0]:.0f},{spot[1]:.0f})")
                else:
                    msg, msg_until = "No free space for a new rectangle", now + 1.5
                    log.event("create refused: no free space")
            elif key == ord('g'):
                grid_on = not grid_on
                log.event(f"grid {'ON' if grid_on else 'OFF'}")
            elif key == ord('u'):
                n = sum(r.locked for r in rects)
                for r in rects:
                    r.locked = False
                log.event(f"unlocked {n} rectangle(s)")

            pobj = selected or flash_obj or hover_obj
            col = worst_collision(pobj, rects) if pobj else None
            info = dict(
                time_s=round(now - t0, 4), state=state, obj=pobj.id if pobj else "-",
                locked=bool(pobj and pobj.locked), hand=hand_status, grid_on=grid_on,
                cx=None, cy=None, w=None, h=None, L=None, R=None, T=None, B=None,
                raw_x=raw[0] if raw else None, raw_y=raw[1] if raw else None,
                cur_x=cursor[0] if cursor else None, cur_y=cursor[1] if cursor else None,
                dist_center=None,
                dx=motion.dx if cursor else None, dy=motion.dy if cursor else None,
                dt=motion.dt if cursor else None,
                vx=motion.vx if cursor else None, vy=motion.vy if cursor else None,
                speed=motion.speed if cursor else None,
                ax=motion.ax if cursor else None, ay=motion.ay if cursor else None,
                acc=motion.acc if cursor else None,
                pinch=pinch_d, pinching=pinching,
                col_other=col[0].id if col else None, col_wi=col[1] if col else None,
                col_hi=col[2] if col else None, col_area=col[3] if col else None,
                col_pct=col[4] if col else None,
                grid_x=None, grid_y=None, snap_x=None, snap_y=None, snap_err=None,
                dwell=dwell_t if selected else None, fps=fps,
            )
            if pobj:
                cx, cy = pobj.posCenter
                L, R, T, B = pobj.bounds
                xs, ys, err = snap_point(cx, cy, GRID_SIZE)
                info.update(cx=cx, cy=cy, w=pobj.size[0], h=pobj.size[1], L=L, R=R, T=T, B=B,
                            grid_x=cx // GRID_SIZE, grid_y=cy // GRID_SIZE,
                            snap_x=xs, snap_y=ys, snap_err=err)
                if cursor:
                    info["dist_center"] = pobj.distance_to(cursor)
            if key == ord('p'):
                log.snapshot(info)
                log.event("snapshot written to lab2_snapshots.csv")
                msg, msg_until = "Snapshot saved", now + 1.0

            if grid_on:
                for gx in range(0, W + 1, GRID_SIZE):
                    for gy in range(0, H + 1, GRID_SIZE):
                        cv2.circle(img, (gx, gy), 1, (90, 90, 90), -1)

            imgNew = np.zeros_like(img, np.uint8)
            for r in rects:
                L, R, T, B = r.bounds
                if r.locked:
                    color = LOCKED_COLOR
                elif r is selected or (r is flash_obj and state in (RELEASED, INVALID)):
                    color = STATE_COLORS[state]
                elif r is hover_obj and state == HOVER:
                    color = STATE_COLORS[HOVER]
                else:
                    color = STATE_COLORS[IDLE]
                cv2.rectangle(imgNew, (int(L), int(T)), (int(R), int(B)), color, cv2.FILLED)
                cvzone.cornerRect(imgNew, (int(L), int(T), int(r.size[0]), int(r.size[1])), 20, rt=0)

            out = img.copy()
            mask = np.any(imgNew > 0, axis=2)
            out[mask] = cv2.addWeighted(img, 0.5, imgNew, 0.5, 0)[mask]

            for r in rects:
                L, R, T, B = r.bounds
                put(out, f"#{r.id}{' LOCK' if r.locked else ''}", (int(L) + 6, int(T) + 20), 0.5)
            if pobj and state != IDLE:
                L, R, T, B = pobj.bounds
                cv2.rectangle(out, (int(L) - 3, int(T) - 3), (int(R) + 3, int(B) + 3), (255, 255, 255), 3)

            if cursor:
                ci = (int(cursor[0]), int(cursor[1]))
                cv2.circle(out, ci, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(out, thumb, 8, (255, 128, 0), cv2.FILLED)
                cv2.line(out, thumb, ci, (0, 255, 0) if pinching else (180, 180, 180), 2)
                if dwell_active and dwell_t > 0.05:
                    cv2.ellipse(out, ci, (24, 24), 0, 0, 360 * min(dwell_t / DWELL_TIME, 1.0),
                                (0, 215, 255), 4)

            if now < msg_until:
                put(out, msg, (W // 2 - 180, 30), 0.7, (0, 215, 255), 2)
            draw_panel(out, info, H)

            if key == ord('s'):
                name = f"lab2_shot_{int(now - t0)}s.png"
                cv2.imwrite(name, out)
                log.event(f"screenshot saved: {name}")

            cv2.imshow("Image", out)
            if key == ord('q'):
                log.event("q pressed: exiting")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
