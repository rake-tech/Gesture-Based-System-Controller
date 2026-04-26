import cv2
import numpy as np
import math
import time
import hand_tracking2 as htm
import screen_brightness_control as sbc
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.HandDetector(detectionCon=0.5)

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Delay control
lastAction = 0
delay = 1

# ---------------- FIST FUNCTION ----------------
def isFist(lmList):
    fingers = []

    fingers.append(lmList[8][2] > lmList[6][2])
    fingers.append(lmList[12][2] > lmList[10][2])
    fingers.append(lmList[16][2] > lmList[14][2])
    fingers.append(lmList[20][2] > lmList[18][2])

    return all(fingers)

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=40)

    img, allHands = detector.findHands(img)

    if len(allHands) != 0:
        for hand in allHands:
            lmList = hand["lmList"]
            handType = hand["type"]

            # Thumb & Index
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1,y1), 12, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 12, (255,0,255), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)

            length = math.hypot(x2-x1, y2-y1)

            # ---------------- PRIORITY: FIST ----------------
            if isFist(lmList):
                if time.time() - lastAction > delay:
                    pyautogui.press('space')
                    lastAction = time.time()

                cv2.putText(img,"PLAY/PAUSE",(200,100),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            # ---------------- NORMAL CONTROLS ----------------
            else:
                # RIGHT HAND → VOLUME
                if handType == "Right":
                    vol = np.interp(length, [30,200], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(vol, None)

                    cv2.putText(img,"RIGHT: Volume",(300,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                # LEFT HAND → BRIGHTNESS
                elif handType == "Left":
                    brightness = np.interp(length, [30,200], [0,100])
                    sbc.set_brightness(int(brightness))

                    cv2.putText(img,"LEFT: Brightness",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    else:
        cv2.putText(img,"NO HAND",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("FINAL AI GESTURE CONTROLLER", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()  