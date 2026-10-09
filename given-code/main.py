import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = None
for camera_index in range(5):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Using camera index {camera_index}")
        break
    cap.release()

if cap is None or not cap.isOpened():
    raise RuntimeError("No webcam found. Please check that your camera is connected and not used by another app.")

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def contains(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        return (cx - w // 2 < cursor[0] < cx + w // 2 and
                cy - h // 2 < cursor[1] < cy + h // 2)

    def update(self, cursor):
        # Move the rectangle to the fingertip (caller decides which rect is active).
        self.posCenter = [int(cursor[0]), int(cursor[1])]


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

activeRect = None  # only one rectangle can be dragged at a time

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame. Trying again...")
        cv2.waitKey(1000)
        continue

    img = cv2.flip(img, 1)
    H, W = img.shape[:2]
    hands, img = detector.findHands(img, draw=False)
    current_index_tip = None

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if len(lmList) >= 21:
            tx = min(max(int(lmList[8][0]), 0), W - 1)
            ty = min(max(int(lmList[8][1]), 0), H - 1)
            current_index_tip = (tx, ty)
            cv2.circle(img, current_index_tip, 10, (0, 255, 0), cv2.FILLED)

            if activeRect is None:
                # Pick the topmost rectangle (last drawn) under the fingertip
                for rect in reversed(rectList):
                    if rect.contains(current_index_tip):
                        activeRect = rect
                        break
            if activeRect is not None:
                activeRect.update(current_index_tip)
    else:
        activeRect = None  # release when the hand disappears

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if current_index_tip is not None:
            rectList.append(DragRect([current_index_tip[0], current_index_tip[1]]))
        else:
            rectList.append(DragRect([W // 2, H // 2]))

    ## Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        cv2.rectangle(imgNew, (x1, y1), (x2, y2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (x1, y1, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    blended = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)
    mask = imgNew.any(axis=2)  # per-pixel mask (H x W), not per-channel
    out[mask] = blended[mask]

    cv2.imshow("Image", out)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


