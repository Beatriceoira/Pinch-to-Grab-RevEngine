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

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Drag the rectangle while the fingertip stays inside its bounds.
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = [int(cursor[0]), int(cursor[1])]


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame. Trying again...")
        cv2.waitKey(1000)
        continue

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=False)
    current_index_tip = None

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if len(lmList) >= 21:
            current_index_tip = tuple(map(int, lmList[8][:2]))
            cv2.circle(img, current_index_tip, 10, (0, 255, 0), cv2.FILLED)
            for rect in rectList:
                rect.update(current_index_tip)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if current_index_tip is not None:
            rectList.append(DragRect([current_index_tip[0], current_index_tip[1]]))
        else:
            rectList.append(DragRect([img.shape[1] // 2, img.shape[0] // 2]))

    ## Draw solid
    # for rect in rectList:
    #     cx, cy = rect.posCenter
    #     w, h = rect.size
    #     cv2.rectangle(img, (cx - w // 2, cy - h // 2),
    #                   (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
    #     cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    ## Draw Transpa2rency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if key == ord('q'):
        break