import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, detectionCon=0.5, maxHands=2):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        allHands = []

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness,
                                         self.results.multi_hand_landmarks):

                lmList = []
                h, w, _ = img.shape

                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                handLabel = handType.classification[0].label

                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

                allHands.append({
                    "lmList": lmList,
                    "type": handLabel
                })

        return img, allHands