import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        # Define drawing styles
        self.poseLandmarkStyle = self.mpDraw.DrawingSpec(color=(0, 100, 0), thickness=8)
        self.poseConnectionStyle = self.mpDraw.DrawingSpec(color=(50, 205, 50), thickness=4)
    
    def findPose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)

        # Draw pose landmarks and connections on the image
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                image, 
                self.results.pose_landmarks, 
                self.mpPose.POSE_CONNECTIONS,
                self.poseLandmarkStyle, 
                self.poseConnectionStyle
            )
        return image
    
    def findPosition(self, image, draw=True):
        """
        Returns a list of landmarks with their coordinates and visibility.
        Each landmark contains x, y, z.
        """
        landmarks  = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return landmarks 