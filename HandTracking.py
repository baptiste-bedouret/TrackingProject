import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, static_image_mode = False, max_num_hands = 2, model_complexity = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.modelComplexity = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        # Define drawing styles
        self.handLandmarkStyle = self.mpDraw.DrawingSpec(color=(0, 100, 0), thickness=8)  # Color and thickness of the landmarks
        self.handConnectionStyle = self.mpDraw.DrawingSpec(color=(50, 205, 50), thickness=4)  # Color and thickness of the connections

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image, 
                        handLms, 
                        self.mpHands.HAND_CONNECTIONS, 
                        self.handLandmarkStyle,  
                        self.handConnectionStyle 
                    )
        return image

    def findPosition(self, image, handNo=0, draw=True):
        landmarks = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return landmarks