from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import threading
import time
import main as m
import HandTracking as htm
import PoseTracking as pt

app = Flask(__name__)

# Global variable to store the selected camera
selected_camera = 0  # Default camera

def get_available_cameras(max_cam=5):
    """Detect and return a list of available cameras."""
    available_cameras = []
    for i in range(max_cam):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

@app.route("/", methods=["GET", "POST"])
def index():
    global selected_camera
    available_cameras = get_available_cameras()

    if request.method == "POST":
        selected_camera = int(request.form["camera"])
        return redirect(url_for("index"))

    return render_template("index.html", cameras=available_cameras, selected_camera=selected_camera)

def generate(selected_camera):
    """Generate video stream from the selected camera."""
    wCam, hCam = 1280, 720
    cap = cv2.VideoCapture(selected_camera)
    cap.set(3, wCam)
    cap.set(4, hCam)

    music_playing = False
    last_dab_time = None
    last_jul_sign_time = None

    hand_detector = htm.HandDetector(min_detection_confidence=0.7)
    pose_detector = pt.PoseDetector()

    while True:
        success, image = cap.read()
        if not success:
            break
        else:
            # Detect pose
            image = pose_detector.findPose(image, draw=False)
            pose_landmarks = pose_detector.findPosition(image, draw=False)

            # Check if a dab is detected
            if m.is_dab(pose_landmarks) and not music_playing:
                last_dab_time = time.time()
                cv2.putText(image, "Dab Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                def play_and_reset_flag():
                    m.play_music_1()
                    nonlocal music_playing
                    music_playing = False
                threading.Thread(target=play_and_reset_flag).start()

            if last_dab_time and time.time() - last_dab_time < 5:
                cv2.putText(image, "Dab Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                last_dab_time = None

            # Detect hands
            image = hand_detector.findHands(image, draw=False)
            hand_landmarks = []
            if hand_detector.results.multi_hand_landmarks:
                for hand_no in range(len(hand_detector.results.multi_hand_landmarks)):
                    hand_landmarks.append(hand_detector.findPosition(image, handNo=hand_no, draw=False))

            # Detect the Jul sign
            if m.is_jul_sign(hand_landmarks):
                last_jul_sign_time = time.time()
                cv2.putText(image, "Jul Sign Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                def play_and_reset_flag():
                    m.play_music_2()
                    nonlocal music_playing
                    music_playing = False
                threading.Thread(target=play_and_reset_flag).start()

            if last_jul_sign_time and time.time() - last_jul_sign_time < 5:
                cv2.putText(image, "Jul Sign Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            else:
                last_jul_sign_time = None

            _, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for the video feed."""
    global selected_camera
    return Response(generate(selected_camera), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))
