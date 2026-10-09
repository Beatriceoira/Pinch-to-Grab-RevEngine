"""
DragRect - pinch to drag squares.

Hover a square, pinch thumb + index to grab it, move, open your fingers to drop.
Squares never overlap: a dragged square stops at another one and slides along it.

Keys: q or x = quit
Run:  python dragrect_drag.py [--camera N]
"""
import argparse
import math
import platform
import sys
import time

import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

FRAME_W, FRAME_H = 1280, 720
GRAB_T = 40.0       # px: pinch closes below this -> grab
RELEASE_T = 70.0    # px: pinch opens above this  -> drop
EMA = 0.6           # cursor smoothing (weight of newest sample)
MAX_DT = 0.25       # s: bigger frame gaps restart smoothing
LOSS_GRACE = 0.4    # s: lost hand keeps its square this long, then drops it
STEP = 10.0         # px: drag moves in steps this small

COL_IDLE, COL_HOVER = (255, 0, 255), (0, 255, 255)
COL_DRAG, COL_BLOCKED = (0, 255, 0), (0, 0, 255)


class Square:
    def __init__(self, cx, cy, size):
        self.cx, self.cy, self.w, self.h = float(cx), float(cy), float(size), float(size)

    @property
    def bounds(self):
        return (self.cx - self.w / 2, self.cx + self.w / 2,
                self.cy - self.h / 2, self.cy + self.h / 2)  # L, R, T, B

    def contains(self, p):
        L, R, T, B = self.bounds
        return L < p[0] < R and T < p[1] < B


def hits(cx, cy, w, h, squares, skip=None):
    L, R, T, B = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    for o in squares:
        if o is skip:
            continue
        oL, oR, oT, oB = o.bounds
        if min(R, oR) - max(L, oL) > 0.01 and min(B, oB) - max(T, oT) > 0.01:
            return True
    return False


def move_toward(sq, tx, ty, squares, W, H):
    """Move sq toward (tx, ty) in small steps; stop/slide at other squares. Returns True if blocked."""
    tx = min(max(tx, sq.w / 2), W - sq.w / 2)
    ty = min(max(ty, sq.h / 2), H - sq.h / 2)
    x, y = sq.cx, sq.cy
    dx, dy = tx - x, ty - y
    n = max(1, math.ceil(max(abs(dx), abs(dy)) / STEP))
    sx, sy = dx / n, dy / n
    stuck = hits(x, y, sq.w, sq.h, squares, sq)
    blocked = False
    for _ in range(n):
        if stuck or not hits(x + sx, y + sy, sq.w, sq.h, squares, sq):
            x, y = x + sx, y + sy
        elif sx and not hits(x + sx, y, sq.w, sq.h, squares, sq):
            x, blocked = x + sx, True
        elif sy and not hits(x, y + sy, sq.w, sq.h, squares, sq):
            y, blocked = y + sy, True
        else:
            blocked = True
            break
    sq.cx, sq.cy = x, y
    return blocked


def pick_target(squares, p):
    best, bd = None, None
    for s in squares:
        if s.contains(p):
            d = math.hypot(p[0] - s.cx, p[1] - s.cy)
            if best is None or d <= bd:
                best, bd = s, d
    return best


def open_camera(only_index=None):
    system = platform.system()
    if system == "Windows":
        apis = ["CAP_DSHOW", "CAP_MSMF", "CAP_ANY"]
    elif system == "Darwin":
        apis = ["CAP_AVFOUNDATION", "CAP_ANY"]
    else:
        apis = ["CAP_V4L2", "CAP_ANY"]
    for index in ([only_index] if only_index is not None else range(5)):
        for attr in apis:
            cap = cv2.VideoCapture(index, getattr(cv2, attr, cv2.CAP_ANY))
            if cap.isOpened():
                cap.set(3, FRAME_W)
                cap.set(4, FRAME_H)
                for _ in range(15):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        print(f"Using camera {index} ({attr})")
                        return cap
                    time.sleep(0.05)
            cap.release()
    return None


def put(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    cap = open_camera(args.camera)
    if cap is None:
        print("ERROR: No webcam found. Check camera permissions / other apps using it. Try --camera N.")
        return 1

    W, H = FRAME_W, FRAME_H
    for _ in range(10):
        ok, probe = cap.read()
        if ok and probe is not None:
            H, W = probe.shape[:2]
            break
    size = min(200, W // 6, H // 3)
    gap = (W - size - 100) / 4
    squares = [Square(size / 2 + 50 + i * gap, size / 2 + 50, size) for i in range(5)]

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    selected, offset = None, (0.0, 0.0)
    pinching = False
    smooth, last_t = None, 0.0
    lost_since = None
    blocked = False
    fails = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if (cv2.waitKey(100) & 0xFF) in (ord('q'), ord('x')) or fails >= 50:
                    break
                continue
            fails = 0
            now = time.time()
            img = cv2.flip(frame, 1)
            H, W = img.shape[:2]

            try:
                hands, img = detector.findHands(img, draw=False)
            except Exception as e:
                print("hand detection error:", e)
                hands = []

            raw, thumb, pinch_d = None, None, None
            if hands:
                lm = hands[0].get("lmList", [])
                if len(lm) >= 21:
                    x8, y8, x4, y4 = lm[8][0], lm[8][1], lm[4][0], lm[4][1]
                    if 0 <= x8 < W and 0 <= y8 < H:
                        raw, thumb = (float(x8), float(y8)), (int(x4), int(y4))
                        pinch_d = math.hypot(x8 - x4, y8 - y4)

            cursor, hover = None, None
            if raw is None:
                smooth = None
                if selected is not None:
                    if lost_since is None:
                        lost_since = now
                    elif now - lost_since > LOSS_GRACE:
                        selected, pinching, lost_since, blocked = None, False, None, False
            else:
                lost_since = None
                if smooth is None or now - last_t > MAX_DT:
                    smooth = raw
                else:
                    smooth = (smooth[0] + EMA * (raw[0] - smooth[0]),
                              smooth[1] + EMA * (raw[1] - smooth[1]))
                last_t = now
                cursor = smooth
                hover = pick_target(squares, cursor)

                was = pinching
                if pinching and pinch_d > RELEASE_T:
                    pinching = False
                elif not pinching and pinch_d < GRAB_T:
                    pinching = True

                if selected is None:
                    if pinching and not was and hover is not None:
                        selected = hover
                        offset = (selected.cx - cursor[0], selected.cy - cursor[1])
                        squares.remove(selected)
                        squares.append(selected)  # draw on top
                elif not pinching:
                    selected, blocked = None, False

                if selected is not None:
                    blocked = move_toward(selected, cursor[0] + offset[0],
                                          cursor[1] + offset[1], squares, W, H)

            # ---- draw
            layer = np.zeros_like(img, np.uint8)
            for s in squares:
                if s is selected:
                    color = COL_BLOCKED if blocked else COL_DRAG
                elif s is hover and selected is None:
                    color = COL_HOVER
                else:
                    color = COL_IDLE
                L, R, T, B = s.bounds
                cv2.rectangle(layer, (int(L), int(T)), (int(R), int(B)), color, cv2.FILLED)
                cvzone.cornerRect(layer, (int(L), int(T), int(s.w), int(s.h)), 20, rt=0)
            out = img.copy()
            mask = np.any(layer > 0, axis=2)
            out[mask] = cv2.addWeighted(img, 0.5, layer, 0.5, 0)[mask]

            if selected is not None:
                L, R, T, B = selected.bounds
                cv2.rectangle(out, (int(L) - 3, int(T) - 3), (int(R) + 3, int(B) + 3),
                              (255, 255, 255), 3)
            if cursor:
                ci = (int(cursor[0]), int(cursor[1]))
                cv2.circle(out, ci, 10, (0, 255, 0), cv2.FILLED)
                if thumb:
                    cv2.circle(out, thumb, 8, (255, 128, 0), cv2.FILLED)
                    cv2.line(out, thumb, ci, (0, 255, 0) if pinching else (180, 180, 180), 2)

            state = ("BLOCKED" if blocked else "DRAGGING") if selected else ("HOVER" if hover else "IDLE")
            put(out, f"STATE: {state}" + (f"   pinch {pinch_d:.0f}px" if pinch_d is not None else "   no hand"),
                (12, H - 40))
            put(out, "pinch to grab | open to drop | q / x = quit", (12, H - 14), 0.5)

            cv2.imshow("Image", out)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('x')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

