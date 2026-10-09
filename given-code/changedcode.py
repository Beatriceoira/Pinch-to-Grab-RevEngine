import argparse
import platform
import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

parser = argparse.ArgumentParser(description="Hover-to-drag rectangles with your index finger")
parser.add_argument("--camera", type=int, default=None,
                    help="camera index to use (default: auto-detect 0-4)")
args = parser.parse_args()


def open_camera(only_index=None):
    """Try the best camera backend for this OS, then fall back to the default."""
    system = platform.system()
    if system == "Windows":
        apis = [("DirectShow", "CAP_DSHOW"), ("Media Foundation", "CAP_MSMF"),
                ("default", "CAP_ANY")]
    elif system == "Darwin":                      # macOS
        apis = [("AVFoundation", "CAP_AVFOUNDATION"), ("default", "CAP_ANY")]
    else:                                         # Linux and others
        apis = [("V4L2", "CAP_V4L2"), ("default", "CAP_ANY")]

    indexes = [only_index] if only_index is not None else range(5)
    for index in indexes:
        for name, attr in apis:
            cap = cv2.VideoCapture(index, getattr(cv2, attr, cv2.CAP_ANY))
            if cap.isOpened():
                cap.set(3, 1280)
                cap.set(4, 720)
                for _ in range(15):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        print(f"Using camera index {index} ({name})")
                        return cap
                    time.sleep(0.05)
            cap.release()
    return None

cap = open_camera(args.camera)
if cap is None:
    hints = {
        "Darwin": "Allow camera access for your terminal/IDE in System Settings > "
                  "Privacy & Security > Camera, then restart it.",
        "Windows": "Check Settings > Privacy > Camera and close other apps using the camera.",
        "Linux": "Check that /dev/video0 exists, your user is in the 'video' group, "
                 "and no other app is using the camera.",
    }
    raise RuntimeError("No webcam found. " + hints.get(platform.system(), "")
                       + " You can also try --camera N.")

detector = HandDetector(detectionCon=0.8, maxHands=1)
colorR = (255, 0, 255)

class DragRect:
    def __init__(self, posCenter, size=(200, 200)):
        self.posCenter = [int(posCenter[0]), int(posCenter[1])]
        self.size = list(size)

    def contains(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        return (cx - w // 2 < cursor[0] < cx + w // 2 and
                cy - h // 2 < cursor[1] < cy + h // 2)

    def distance(self, cursor):
        cx, cy = self.posCenter
        return ((cursor[0] - cx) * 2 + (cursor[1] - cy) * 2) ** 0.5

    def moveTo(self, cursor, frame_w, frame_h):
        w, h = self.size
        self.posCenter = [min(max(int(cursor[0]), w // 2), frame_w - w // 2),
                          min(max(int(cursor[1]), h // 2), frame_h - h // 2)]


def pick_target(rects, cursor):
    """FIX: only ONE rectangle moves - the nearest centre under the cursor
    (ties go to the topmost). Before, every overlapping rectangle moved together."""
    best, best_d = None, None
    for r in rects:
        if r.contains(cursor):
            d = r.distance(cursor)
            if best is None or d <= best_d:
                best, best_d = r, d
    return best


rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]
fails = 0

try:
    while True:
        success, img = cap.read()
        if not success:
            fails += 1
            print(f"Failed to grab frame ({fails}). Trying again...")
            k = cv2.waitKey(100) & 0xFF
            if k in (ord('q'), ord('x')) or fails >= 50:
                break
            continue
        fails = 0

        img = cv2.flip(img, 1)
        H, W = img.shape[:2]
        hands, img = detector.findHands(img, draw=False)
        current_index_tip = None

        if hands:
            lmList = hands[0]["lmList"]
            if len(lmList) >= 21:
                x, y = int(lmList[8][0]), int(lmList[8][1])
                if 0 <= x < W and 0 <= y < H:
                    current_index_tip = (x, y)

        if current_index_tip is not None:
            cv2.circle(img, current_index_tip, 10, (0, 255, 0), cv2.FILLED)
            target = pick_target(rectList, current_index_tip)
            if target is not None:
                target.moveTo(current_index_tip, W, H)
                rectList.remove(target)
                rectList.append(target)          # draw the dragged one on top

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if current_index_tip is not None:
                rectList.append(DragRect(current_index_tip))
            else:
                rectList.append(DragRect([W // 2, H // 2]))

        imgNew = np.zeros_like(img, np.uint8)
        for rect in rectList:
            cx, cy = rect.posCenter
            w, h = rect.size
            cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                          (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
            cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

        out = img.copy()
        alpha = 0.5
        mask = np.any(imgNew > 0, axis=2)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

        cv2.imshow("Image", out)
        if key in (ord('q'), ord('x')):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
