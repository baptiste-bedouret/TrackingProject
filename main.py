import cv2
import mediapipe as mp
import time
from playsound import playsound
import threading
import HandTracking as htm
import PoseTracking as pt
import math
import pygame


def calculate_angle(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # Prevent division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    return math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))


def dab_left(landmarks):

    # Ensure all landmarks are available
    if len(landmarks) >= 33:

        # Right side landmarks
        right_elbow = landmarks[13][1:3]

        # Left side landmarks
        left_shoulder = landmarks[12][1:3]
        left_elbow = landmarks[14][1:3]
        left_wrist = landmarks[16][1:3]

        # Head (nose landmark)
        head = landmarks[0][1:3]

        # Compute vectors
        shoulder_to_elbow = (left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1])
        elbow_to_wrist = (left_wrist[0] - left_elbow[0], left_wrist[1] - left_elbow[1])

        # Angle between shoulder-to-elbow and elbow-to-wrist (should be close to 180 for a straight arm)
        left_elbow_angle = calculate_angle(shoulder_to_elbow, elbow_to_wrist)

        # Ensure the left arm is extended (elbow angle close to 0 degrees)
        left_arm_extended = 0 <= left_elbow_angle <= 30

        # Compute the vector for the left arm (shoulder to wrist) for horizontal alignment
        left_arm_vector = (left_wrist[0] - left_shoulder[0], left_wrist[1] - left_shoulder[1])
        left_arm_angle = math.degrees(math.atan2(-left_arm_vector[1], left_arm_vector[0]))

        # Check if the left arm is nearly horizontal
        left_arm_near_horizontal = 5 <= left_arm_angle <= 40

        # Check if the head is near the right elbow
        head_near_right_elbow = abs(head[0] - right_elbow[0]) < 150 and abs(head[1] - right_elbow[1]) < 150

        # Dab detection
        dab_left = left_arm_extended and left_arm_near_horizontal and head_near_right_elbow

    return dab_left


def dab_right(landmarks):

    # Ensure all landmarks are available
    if len(landmarks) >= 33:
        # Left side landmarks
        left_elbow = landmarks[14][1:3]

        # Right side landmarks
        right_shoulder = landmarks[11][1:3]
        right_elbow = landmarks[13][1:3]
        right_wrist = landmarks[15][1:3]

        # Head (nose landmark)
        head = landmarks[0][1:3]

        # Compute vectors
        shoulder_to_elbow = (right_elbow[0] - right_shoulder[0], left_elbow[1] - right_shoulder[1])
        elbow_to_wrist = (right_wrist[0] - right_elbow[0], right_wrist[1] - right_elbow[1])

        # Angle between shoulder-to-elbow and elbow-to-wrist (should be close to 180 for a straight arm)
        right_elbow_angle = calculate_angle(shoulder_to_elbow, elbow_to_wrist)

        # Ensure the left arm is extended (elbow angle close to 0 degrees)
        right_arm_extended = 0 <= right_elbow_angle <= 30

        # Compute the vector for the right arm (shoulder to wrist) for horizontal alignment
        right_arm_vector = (right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1])
        right_arm_angle = math.degrees(math.atan2(-right_arm_vector[1], right_arm_vector[0]))

        # Check if the left arm is nearly horizontal
        right_arm_near_horizontal = 5 <= right_arm_angle <= 40

        # Check if the head is near the left elbow
        head_near_left_elbow = abs(head[0] - left_elbow[0]) < 150 and abs(head[1] - left_elbow[1]) < 150

        # Dab detection
        dab_right = right_arm_extended and right_arm_near_horizontal and head_near_left_elbow

    return dab_right


def is_dab(landmarks):
    """
    Detect if the user is performing a dab.
    Returns True if a dab is detected, otherwise False.
    """
    if dab_right(landmarks):
        return True
    return False


def play_music_1():
    """Plays the music file and waits for it to finish."""
    pygame.mixer.init()
    pygame.mixer.music.load("music/MCFLY & CARLITO - JEFFECTUE LE DAB.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Check every 100ms
        

def is_jul_sign(landmarks):
    """
    Detect if the user is performing the Jul sign with one or both hands.
    Returns True if the Jul sign is detected.
    """
    # Wait until landmarks is not None
    if landmarks is None or len(landmarks) == 0:
        time.sleep(0.1) 

    jul_hands = []
    if len(landmarks) == 2:
        # print("Number of hands detected:", len(landmarks))
        landmarks_0 = landmarks[0]
        landmarks_1 = landmarks[1]
        
        # Get positions of relevant landmarks
        thumb_tip_0 = landmarks_0[4]
        index_tip_0 = landmarks_0[8]
        middle_tip_0 = landmarks_0[12]
        ring_tip_0 = landmarks_0[16]
        pinky_tip_0 = landmarks_0[20] 

        thumb_tip_1 = landmarks_1[4]
        index_tip_1 = landmarks_1[8]
        middle_tip_1 = landmarks_1[12]
        ring_tip_1 = landmarks_1[16]
        pinky_tip_1 = landmarks_1[20]   

        # Conditions for Jul sign:
        pinky_bent_0 = pinky_tip_0[1] > landmarks_0[19][1]  # Pinky below DIP joint
        ring_bent_0 = ring_tip_0[1] > landmarks_0[15][1]  # Ring finger below DIP joint
        thumb_extended_0 = thumb_tip_0[1] < landmarks_0[2][1]  # Thumb above MCP joint
        index_extended_0 = index_tip_0[1] < landmarks_0[7][1]  # Index finger above DIP joint
        middle_extended_0 = middle_tip_0[1] < landmarks_0[11][1]  # Middle finger above DIP joint

        pinky_bent_1 = pinky_tip_1[1] > landmarks_1[19][1]  # Pinky below DIP joint
        ring_bent_1 = ring_tip_1[1] > landmarks_1[15][1]  # Ring finger below DIP joint
        thumb_extended_1 = thumb_tip_1[1] < landmarks_1[2][1]  # Thumb above MCP joint
        index_extended_1 = index_tip_1[1] < landmarks_1[7][1]  # Index finger above DIP joint
        middle_extended_1 = middle_tip_1[1] < landmarks_1[11][1]  # Middle finger above DIP joint

        # Check if hand 1 performs the Jul sign
        if thumb_extended_0 and pinky_bent_0 and index_extended_0 and middle_extended_0 and ring_bent_0:
            jul_hands.append(True)
        else:
            jul_hands.append(False)

        # Check if hand 2 performs the Jul sign
        if thumb_extended_1 and pinky_bent_1 and index_extended_1 and middle_extended_1 and ring_bent_1:
            jul_hands.append(True)
        else:
            jul_hands.append(False)

        # If both hands perform the Jul sign or one hand performs the Jul sign, return True
        if any(jul_hands):
            return True
    return False


def play_music_2():
    """Plays the music file and waits for it to finish."""
    pygame.mixer.init()
    pygame.mixer.music.load("music/Jul - JCVD  Clip Officiel  2019.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Check every 100ms


# Dimensions of the camera feed
wCam, hCam = 1280, 720

def main():

    # Open the camera 
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    # Hand detector instance
    hand_detector = htm.HandDetector(min_detection_confidence = 0.7)

    # Pose detector instance
    pose_detector = pt.PoseDetector()

    music_playing = False

    last_dab_time = None
    last_jul_sign_time = None

    while True:
        # Read the image from the camera
        success, image = cap.read()

        # Detect pose
        image = pose_detector.findPose(image)    
        pose_landmarks = pose_detector.findPosition(image)

        # Check if a dab is detected
        if is_dab(pose_landmarks) and not music_playing:
            last_dab_time = time.time()
            cv2.putText(image, "Dab Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # print("Dab Detected!")

            # Play music in a separate thread to avoid blocking
            def play_and_reset_flag():
                play_music_1()
                nonlocal music_playing
                music_playing = False  # Reset the flag when music finishes
            threading.Thread(target=play_and_reset_flag).start()

        # Keep displaying the text for 5 seconds after the last detection
        if last_dab_time and time.time() - last_dab_time < 5:
            cv2.putText(image, "Dab Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            # Clear the timestamp after 5 seconds
            last_dab_time = None

        # Detect hands
        image = hand_detector.findHands(image)
        hand_landmarks = []
        if hand_detector.results.multi_hand_landmarks:
            for hand_no in range(len(hand_detector.results.multi_hand_landmarks)):
                hand_landmarks.append(hand_detector.findPosition(image, handNo=hand_no, draw=False))

        # Detect the Jul sign
        if is_jul_sign(hand_landmarks):
            last_jul_sign_time = time.time()
            cv2.putText(image, "Jul Sign Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            # print("Jul Sign Detected!")

            # Play music in a separate thread to avoid blocking
            def play_and_reset_flag():
                play_music_2()
                nonlocal music_playing
                music_playing = False  # Reset the flag when music finishes
            threading.Thread(target=play_and_reset_flag).start()

        # Keep displaying the text for 5 seconds after the last detection
        if last_jul_sign_time and time.time() - last_jul_sign_time < 5:
            cv2.putText(image, "Jul Sign Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        else:
            # Clear the timestamp after 5 seconds
            last_jul_sign_time = None

        # Print the image
        cv2.imshow("Pose tracking", image)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Free up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()