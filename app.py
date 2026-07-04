# ============================================================
# Driver Drowsiness Detection System
# ============================================================
# This application uses a webcam or a video file to analyze
# the driver's state in real time, and issues an alert when
# it detects drowsiness or frequent yawning.
# ============================================================

# ── Libraries ────────────────────────────────────────────────
import cv2                          # Image and video processing (OpenCV)
import dlib                         # Face detection and facial landmark detection
import numpy as np                  # Array and matrix operations
from scipy.spatial import distance  # Euclidean distance calculations between points
from imutils.video import VideoStream  # Video stream acquisition from camera
from imutils import face_utils      # Convert facial landmarks to numpy arrays
from dataclasses import dataclass   # Data structure for configuration (Config)
from flask import Flask, Response, render_template_string, request, session  # Web server
import time                         # Time measurement for alert timers
import winsound                     # Play default Windows beep (fallback)
import os                           # File and folder operations
from werkzeug.utils import secure_filename  # Secure uploaded file names
import datetime
from collections import deque       # Double-ended queue for storing yawn timestamps
import pygame                       # Play custom audio file (MP3)

# ── Initialize Pygame audio engine ──────────────────────────
# Initialize Pygame audio mixer for custom sound playback
pygame.mixer.init()

# ── Flask (web server) configuration ────────────────────────
# Flask app setup: secret key for sessions, upload folder config
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'   # Session secret key (change in a real project)
UPLOAD_FOLDER = 'uploads'                  # Folder for storing uploaded videos
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}  # Allowed video formats


def allowed_file(filename):
    """
    Check if uploaded file has an allowed video extension.
    Example: 'video.mp4' -> True  |  'image.jpg' -> False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------------------------------------------
# Event Logging System
# -----------------------------------------------------------------
LOG_FILE = "logs/drowsiness.log"  # Log file path


def get_datetime():
    """
    Returns current date and time in Gregorian calendar format.
    Example output: '2026/04/06 - 15:45:12'
    """
    now = datetime.datetime.now()
    return now.strftime("%Y/%m/%d - %H:%M:%S")

def log_event(event_type):
    """
    Logs a drowsiness or yawn alert event with a Persian timestamp to a text file.

    args:
        event_type: type of event
            'eye_closed_5s'  -> eyes were closed for more than 5 seconds
            'frequent_yawn'  -> more than 3 yawns within 2 minutes
    """
    timestamp = get_datetime()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if event_type == "eye_closed_5s":
            f.write(f"[{timestamp}] ALERT: Eyes closed for more than 5 seconds (drowsiness risk)\n")
        elif event_type == "frequent_yawn":
            f.write(f"[{timestamp}] ALERT: More than 3 yawns within 2 minutes (sign of fatigue)\n")

    # Print to console for debugging
    print(f"LOG: {event_type} at {timestamp}")


# -----------------------------------------------------------------
# Project Configuration - All thresholds in one place
# -----------------------------------------------------------------
@dataclass
class Config:
    """
    All tunable thresholds and parameters in a single dataclass.

    Attributes:
        EYE_AR_THRESH          : EAR threshold - below this value, the eye is considered closed
        EYE_AR_CONSEC_FRAMES   : number of consecutive frames with closed eyes to confirm drowsiness
        YAWN_THRESH            : lip-distance threshold for yawn detection (in pixels)
        YAWN_CONSEC_FRAMES     : number of consecutive frames to confirm a yawn
        DROWSY_TIME_SECONDS    : duration (in seconds) of closed eyes required to trigger the alert sound
        CUSTOM_ALERT_SOUND_PATH: path to the custom alert sound file
    """
    EYE_AR_THRESH: float = 0.25
    EYE_AR_CONSEC_FRAMES: int = 20
    YAWN_THRESH: float = 29
    YAWN_CONSEC_FRAMES: int = 7
    DROWSY_TIME_SECONDS: float = 4.0
    CUSTOM_ALERT_SOUND_PATH: str = "sound/alert.mp3"


# -----------------------------------------------------------------
# Facial Feature Extraction using dlib 68-point landmarks
# -----------------------------------------------------------------
class FacialFeatures:
    """
    Detects faces and computes geometric metrics (EAR, lip distance) using dlib.
    """

    def __init__(self):
        # Initialize face detector and 68-point facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    def calculate_eye_aspect_ratio(self, eye_points):
        """
        Computes Eye Aspect Ratio (EAR) from 6 eye landmark points.

        Formula: EAR = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)

        Value near 0    -> eye closed
        Value near 0.3+ -> eye open

        1e-6 is added to the denominator to prevent division by zero.
        """
        v1 = distance.euclidean(eye_points[1], eye_points[5])  # First vertical distance
        v2 = distance.euclidean(eye_points[2], eye_points[4])  # Second vertical distance
        h = distance.euclidean(eye_points[0], eye_points[3])   # Horizontal distance
        return (v1 + v2) / (2.0 * h + 1e-6)

    def get_eye_measurements(self, shape):
        """
        Measures EAR for both eyes and returns the minimum value.

        Returns: (min_ear, left_eye_points, right_eye_points)

        Note: The minimum EAR is used - if either eye closes, the alert should trigger.
        """
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.calculate_eye_aspect_ratio(leftEye)
        rightEAR = self.calculate_eye_aspect_ratio(rightEye)
        return (min(leftEAR, rightEAR), leftEye, rightEye)

    def calculate_lip_distance(self, shape):
        """
        Computes the vertical distance between the upper and lower lip (yawn detection metric).

        Landmark indices used from the dlib 68-point model:
            Upper lip: points 50-52 and 61-63
            Lower lip: points 56-58 and 65-67

        Larger distance -> mouth more open -> likely yawning
        """
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        bottom_lip = shape[56:59]
        bottom_lip = np.concatenate((bottom_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        bottom_mean = np.mean(bottom_lip, axis=0)
        return abs(top_mean[1] - bottom_mean[1])  # Difference in y-coordinates


# -----------------------------------------------------------------
# Main Drowsiness Detection Engine
# -----------------------------------------------------------------
class DrowsinessDetector:
    """
    Main engine: processes each video frame and manages alert states.
    """

    def __init__(self):
        self.config = Config()              # Load configuration
        self.facial_features = FacialFeatures()  # Initialize the face detector

        # ── Counters ─────────────────────────────────────────
        self.eye_counter = 0    # Number of consecutive frames with closed eyes
        self.yawn_counter = 0   # Number of consecutive frames with a yawn

        # ── Current States ───────────────────────────────────
        self.is_drowsy = False
        self.is_yawning = False

        # ── History for smoothing ────────────────────────────
        # Keep the last 5 EAR and lip-distance values for noise reduction
        self.ear_history = []
        self.lip_history = []

        # ── Eye-closed timer ─────────────────────────────────
        self.eye_closed_start_time = None   # Timestamp when the eyes started closing
        self.eye_closed_logged = False      # Whether this event has already been logged (avoid duplicates)

        # ── Yawn frequency tracking ──────────────────────────
        # deque stores yawn timestamps within the last 2-minute window
        self.yawn_timestamps = deque()
        self.frequent_yawn_logged = False   # Whether the "frequent yawning" alert has already been logged

        # ── Alert persistence ────────────────────────────────
        # Alert stays visible for 3 seconds after the condition clears (prevents flickering)
        self.last_drowsy_time = 0
        self.last_yawn_time = 0

    def reset_state(self):
        """
        Resets all internal state - called when switching video source or restarting.
        """
        self.eye_counter = 0
        self.yawn_counter = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.ear_history = []
        self.lip_history = []
        self.eye_closed_start_time = None
        self.eye_closed_logged = False
        self.yawn_timestamps.clear()
        self.frequent_yawn_logged = False
        self.last_drowsy_time = 0
        self.last_yawn_time = 0

    def play_custom_sound(self):
        """
        Plays a custom MP3 alert; falls back to a Windows Beep if the file is not found or an error occurs.
        """
        try:
            winsound.Beep(1000, 50)  # Short initial beep
            pygame.mixer.music.load(self.config.CUSTOM_ALERT_SOUND_PATH)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Sound error: {e}. Using fallback beep.")
            winsound.Beep(1000, 500)  # Default fallback long beep

    def process_frame(self, frame):
        """
        Core method: processes each video frame for drowsiness/yawn detection.

        Processing steps:
        1. Resize to 640x480 and convert to grayscale
        2. Detect faces with dlib
        3. Extract facial landmarks and compute EAR and lip distance
        4. Average over the last 5 frames to reduce noise
        5. Evaluate drowsiness and yawn conditions
        6. Draw contours around the eyes and mouth
        """
        if frame is None:
            return frame

        # ── Frame preprocessing ──────────────────────────────
        frame = cv2.resize(frame, (640, 480))       # Standardize size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray = cv2.equalizeHist(gray)               # Improve contrast for better detection

        if gray is None or gray.size == 0:
            return frame

        # Ensure memory contiguity required by dlib
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        faces = self.facial_features.detector(gray, 0)  # Detect faces

        current_time = time.time()

        # ── Remove yawn timestamps older than 2 minutes ──────
        # Remove yawn timestamps older than 2 minutes (120 seconds)
        while self.yawn_timestamps and current_time - self.yawn_timestamps[0] > 120:
            self.yawn_timestamps.popleft()

        for face in faces:
            # Extract 68 facial landmarks
            shape = self.facial_features.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # ── Calculate EAR and maintain rolling average ────
            # Calculate EAR and maintain rolling average over last 5 frames
            ear, leftEye, rightEye = self.facial_features.get_eye_measurements(shape)
            self.ear_history.append(ear)
            if len(self.ear_history) > 5:
                self.ear_history.pop(0)

            # ── Calculate lip distance and maintain rolling average ─
            # Calculate lip distance and maintain rolling average over last 5 frames
            lip_distance = self.facial_features.calculate_lip_distance(shape)
            self.lip_history.append(lip_distance)
            if len(self.lip_history) > 5:
                self.lip_history.pop(0)

            avg_ear = np.mean(self.ear_history)
            avg_lip_distance = np.mean(self.lip_history)

            #  -----------------------------------------------------------------
            # Section 1: Drowsiness Detection (eye closed > 4 seconds)
            #  -----------------------------------------------------------------
            if avg_ear < self.config.EYE_AR_THRESH:
                # Start eye-closed timer if not already started
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = current_time

                closed_duration = current_time - self.eye_closed_start_time

                # After 2s of closure, increment frame counter faster
                if closed_duration >= 2.0:
                    self.eye_counter += 2
                    if self.eye_counter >= self.config.EYE_AR_CONSEC_FRAMES:
                        self.is_drowsy = True
                        self.last_drowsy_time = current_time
                else:
                    self.is_drowsy = False

                # After DROWSY_TIME_SECONDS: play alert sound and log once
                if closed_duration >= self.config.DROWSY_TIME_SECONDS:
                    self.play_custom_sound()
                    if not self.eye_closed_logged:   # Log only once
                        log_event("eye_closed_5s")
                        self.eye_closed_logged = True

                self.eye_counter += 2
                if self.eye_counter >= self.config.EYE_AR_CONSEC_FRAMES:
                    self.is_drowsy = True
                    self.last_drowsy_time = current_time

            else:
                # ── Eye opened: gradually decrease counter ────
                # Eye opened: gradually decrease counter (hysteresis to avoid flickering)
                self.eye_counter = max(0, self.eye_counter - 1)
                if self.eye_counter < self.config.EYE_AR_CONSEC_FRAMES:
                    self.is_drowsy = False

                # Reset closed-eye timer when eye opens
                if self.eye_closed_start_time is not None:
                    self.eye_closed_start_time = None
                    self.eye_closed_logged = False

            #  -----------------------------------------------------------------
            # Section 2: Yawn Detection and Frequency Check
            # -----------------------------------------------------------------
            if avg_lip_distance > self.config.YAWN_THRESH:
                self.yawn_counter += 2
                if self.yawn_counter >= self.config.YAWN_CONSEC_FRAMES:
                    self.is_yawning = True
                    self.last_yawn_time = current_time

                    # Record yawn timestamp - throttled to once per second to avoid duplicates
                    if not hasattr(self, 'last_yawn_record_time') or \
                       (current_time - getattr(self, 'last_yawn_record_time', 0) > 1.0):
                        self.yawn_timestamps.append(current_time)
                        self.last_yawn_record_time = current_time

                        # If more than 3 yawns in 2 minutes -> log fatigue alert
                        if len(self.yawn_timestamps) > 3 and not self.frequent_yawn_logged:
                            log_event("frequent_yawn")
                            self.frequent_yawn_logged = True
            else:
                # ── Gradually decrease yawn counter ───────────
                # Gradually decrease yawn counter when mouth closes
                self.yawn_counter = max(0, self.yawn_counter - 1)
                if self.yawn_counter < self.config.YAWN_CONSEC_FRAMES:
                    self.is_yawning = False

                # Reset frequent-yawn flag if count drops back to 3 or below
                if len(self.yawn_timestamps) <= 3:
                    self.frequent_yawn_logged = False

            # -----------------------------------------------------------------
            # Section 3: Draw contours on eyes and mouth
            #  -----------------------------------------------------------------
            # Red = alert state, Green = normal state
            eye_color = (0, 0, 255) if avg_ear < self.config.EYE_AR_THRESH else (0, 255, 0)
            lip_color = (0, 0, 255) if avg_lip_distance > self.config.YAWN_THRESH else (0, 255, 0)

            # Draw convex hull around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, eye_color, 2)
            cv2.drawContours(frame, [rightEyeHull], -1, eye_color, 2)
            cv2.drawContours(frame, [shape[48:60]], -1, lip_color, 2)  # Mouth contour

        return frame

    def get_alert_states(self):
        """
        Returns current alert states with 3-second persistence.

        Alert remains True for 3 seconds after the condition clears (prevents flickering).

        Returns: (is_drowsy: bool, is_yawning: bool)
        """
        now = time.time()
        drowsy = self.is_drowsy or (now - self.last_drowsy_time <= 3.0)
        yawning = self.is_yawning or (now - self.last_yawn_time <= 3.0)
        return drowsy, yawning


# -----------------------------------------------------------------
# Flask App Initialization and Global Resources
# -----------------------------------------------------------------
detector = DrowsinessDetector()      # Main drowsiness detector instance
cap = cv2.VideoCapture(0)            # Connect to the webcam (index 0 = default camera)
uploaded_video_path = None           # Path of the uploaded video (if any)
uploaded_cap = None                  # VideoCapture object for the uploaded video

# -----------------------------------------------------------------
# HTML Template - Web User Interface
# -----------------------------------------------------------------
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html dir="ltr">
<head>
    <meta charset="UTF-8">
    <title>Driver Drowsiness Monitoring System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; color: #1a237e; }
        .video-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
        }
        .video-feed { width: 100%; max-width: 640px; height: auto; border-radius: 10px; display: block; }

        /* Alerts - hidden by default, shown/hidden via JavaScript */
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            display: none;
            animation: blink 1s infinite;
        }
        @keyframes blink { 50% { opacity: 0.5; } }
        #drowsinessAlert { background-color: #ff5252; color: white; }
        #yawnAlert { background-color: #ff9800; color: white; }

        .button-group { text-align: center; margin: 20px; }
        button {
            background-color: #1a237e;
            border: none;
            color: white;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background-color: #0d1652; }
        #uploadInput { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Driver Monitoring System</h1>
            <h3>Drowsiness and Yawn Detection</h3>
        </div>
        <div class="button-group">
            <button id="webcamBtn">📷 Webcam</button>
            <button id="uploadBtn">🎥 Upload Video</button>
            <input type="file" id="uploadInput" accept="video/*">
        </div>
        <div class="video-container">
            <!-- Video stream from Flask server -->
            <img id="videoFeed" src="{{ url_for('video_feed') }}" class="video-feed">
            <div id="drowsinessAlert" class="alert">⚠️ Alert: Drowsiness detected - please take a break ⚠️</div>
            <div id="yawnAlert" class="alert">😴 Frequent yawning - high fatigue risk 😴</div>
        </div>
    </div>

    <script>
        let currentStream = 'webcam';

        /**
         * Polls /check_alerts every 300ms and shows/hides alert banners.
         */
        function checkAlerts() {
            fetch('/check_alerts')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('drowsinessAlert').style.display =
                        data.is_drowsy ? 'block' : 'none';
                    document.getElementById('yawnAlert').style.display =
                        data.is_yawning ? 'block' : 'none';
                });
        }
        setInterval(checkAlerts, 300);  // Poll every 300ms

        /* Webcam button: switch to live camera */
        document.getElementById('webcamBtn').onclick = function() {
            currentStream = 'webcam';
            document.getElementById('videoFeed').src = "{{ url_for('video_feed') }}?" + new Date().getTime();
        };

        /* Upload button: open file picker */
        document.getElementById('uploadBtn').onclick = function() {
            document.getElementById('uploadInput').click();
        };

        /* After file selected: upload to Flask server and switch stream */
        document.getElementById('uploadInput').onchange = function(event) {
            const file = event.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('video', file);
            fetch('/upload_video', {
                method: 'POST',
                body: formData
            }).then(response => {
                if (response.ok) {
                    document.getElementById('videoFeed').src =
                        "{{ url_for('video_feed_upload') }}?" + new Date().getTime();
                } else {
                    alert('Upload failed');
                }
            });
        };
    </script>
</body>
</html>
'''

# -----------------------------------------------------------------
# Flask Route Handlers
# -----------------------------------------------------------------

@app.route('/')
def index():
    """Main page - reset state and render UI"""
    detector.reset_state()
    return render_template_string(HTML_TEMPLATE)


def generate_frames_webcam():
    """
    Generates processed frames from the webcam as an MJPEG stream for browser display.
    """
    global cap
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = detector.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        # Multipart format for browser streaming
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_frames_upload():
    """
    Generates processed frames from the uploaded video file as an MJPEG stream.
    """
    global uploaded_cap
    if uploaded_cap is None:
        return
    while True:
        ret, frame = uploaded_cap.read()
        if not ret or frame is None:
            break
        frame = detector.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Webcam video stream endpoint"""
    return Response(generate_frames_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_upload')
def video_feed_upload():
    """Uploaded video stream endpoint"""
    return Response(generate_frames_upload(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Saves the uploaded video file and opens it for streaming.

    Validations performed:
    - File present in request
    - File actually selected
    - Allowed extension (mp4, avi, mov, mkv)
    """
    global uploaded_cap, uploaded_video_path
    if 'video' not in request.files:
        return 'No file part', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Sanitize filename for security
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Release previous VideoCapture to prevent resource leak
        if uploaded_cap is not None:
            uploaded_cap.release()
        uploaded_cap = cv2.VideoCapture(filepath)
        uploaded_video_path = filepath
        detector.reset_state()  # Reset state for the new video
        return 'OK', 200
    return 'Invalid file type', 400


@app.route('/check_alerts')
def check_alerts():
    """
    Returns current alert states as JSON - polled every 300ms by the browser.

    Sample response:
        {"is_drowsy": false, "is_yawning": true}
    """
    drowsy, yawning = detector.get_alert_states()
    return {"is_drowsy": drowsy, "is_yawning": yawning}


# -----------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------
if __name__ == '__main__':
    try:
        # Run Flask server on port 5000 (access at http://localhost:5000)
        app.run(debug=False, port=5000)
    finally:
        # Release camera resources on exit to prevent hardware lock
        if cap is not None:
            cap.release()
        if uploaded_cap is not None:
            uploaded_cap.release()
        cv2.destroyAllWindows()
