import cv2
from cvzone.HandTrackingModule import HandDetector
import math

#camera
cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,250)

#hand detect
detector = HandDetector(detectionCon=0.8, maxHands=1)


while True:
    success, img = cap.read()
    hand, img = detector.findHands(img)

    if hand:
        point_index = hand[0]['lmList']
        print(point_index)
        #distance of the hand
        x1, y1, z1 = point_index[5]
        x2, y2, z2 = point_index[17]
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)

        print(distance)
    cv2.imshow("Image",img)
    cv2.waitKey(1)