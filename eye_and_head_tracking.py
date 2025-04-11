import cv2
import dlib
import numpy as np

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
NOSE = list(range(27, 31))

def get_eye_data(eye_points, frame):
    (x, y, w, h) = cv2.boundingRect(np.array(eye_points))
    eye_img = frame[y:y+h, x:x+w]
    eye_center = (x + w//2, y + h//2)
    return eye_img, (x, y, w, h), eye_center

def get_pupil_center(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def estimate_gaze_angle(pupil_center, eye_box):
    px, _ = pupil_center
    _, _, w, _ = eye_box
    center_x = w / 2
    relative_offset = (px - center_x) / w
    angle = relative_offset * 30  # Assuming ±30° horizontal FOV
    return angle

def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

# Webcam stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye_pts = [landmarks[i] for i in LEFT_EYE]
        right_eye_pts = [landmarks[i] for i in RIGHT_EYE]

        # Eye centers
        left_eye_img, left_box, left_center = get_eye_data(left_eye_pts, frame)
        right_eye_img, right_box, right_center = get_eye_data(right_eye_pts, frame)

        # Pupil detection
        left_pupil = get_pupil_center(left_eye_img)
        right_pupil = get_pupil_center(right_eye_img)

        if left_pupil and right_pupil:
            left_angle = estimate_gaze_angle(left_pupil, left_box)
            right_angle = estimate_gaze_angle(right_pupil, right_box)

            # Draw pupils
            cv2.circle(left_eye_img, left_pupil, 3, (255, 255, 0), -1)
            cv2.circle(right_eye_img, right_pupil, 3, (255, 255, 0), -1)

            # Display angles
            cv2.putText(frame, f"Left Eye: {left_angle:+.1f}°", (left_box[0], left_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Right Eye: {right_angle:+.1f}°", (right_box[0], right_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Head tilt angle
        left_mid = midpoint(landmarks[36], landmarks[39])
        right_mid = midpoint(landmarks[42], landmarks[45])
        dx = right_mid[0] - left_mid[0]
        dy = right_mid[1] - left_mid[1]
        tilt_angle = np.degrees(np.arctan2(dy, dx))

        cv2.line(frame, left_mid, right_mid, (0, 0, 255), 2)
        cv2.putText(frame, f"Head Tilt: {tilt_angle:+.1f}°", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw green dots on face landmarks (optional)
        for pt in LEFT_EYE + RIGHT_EYE + NOSE:
            cv2.circle(frame, landmarks[pt], 2, (0, 255, 0), -1)

    cv2.imshow("Gaze Angle & Head Tilt Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
