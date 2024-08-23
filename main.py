import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Initialize VideoCapture
cam = cv2.VideoCapture(0)

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Define the neutral gaze calibration duration
calibration_duration = 5  # seconds
calibration_start_time = time.time()
calibration_positions = []

# Movement configuration
move_speed = 10  # Pixels to move per step
smoothing_factor = 0.2  # For smoothing the movement

# Stability configuration
stability_threshold = 20  # Pixels within which gaze is considered stable
stability_time_threshold = 0.5  # Seconds the gaze must be stable to hold the cursor
last_stable_time = time.time()  # Time when the gaze was last stable

# Initialize cursor position
current_screen_x, current_screen_y = screen_width // 2, screen_height // 2

# Function to calculate the iris center for left and right eyes
def get_iris_center(landmarks, frame_width, frame_height, indices):
    x = int(sum([landmarks[i].x * frame_width for i in indices]) / len(indices))
    y = int(sum([landmarks[i].y * frame_height for i in indices]) / len(indices))
    return x, y

# Iris landmark indices for the left eye
left_iris_indices = [474, 475, 476, 477]
# Iris landmark indices for the right eye
right_iris_indices = [145, 159]

# Eyelid landmark indices for the right eye
right_eye_upper_indices = [159]
right_eye_lower_indices = [145]

# Threshold for detecting a wink (can be adjusted)
wink_threshold = 9

# Initial neutral gaze values
neutral_gaze_x, neutral_gaze_y = 0, 0

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with FaceMesh
    output = face_mesh.process(rgb_frame)

    # Get landmark points if available
    detected_landmarks = output.multi_face_landmarks
    frame_height, frame_width, _ = frame.shape

    if detected_landmarks:
        for face_landmarks in detected_landmarks:
            # Get the center of the left iris (or right iris if you prefer)
            left_iris_center = get_iris_center(face_landmarks.landmark, frame_width, frame_height, left_iris_indices)
            right_iris_center = get_iris_center(face_landmarks.landmark, frame_width, frame_height, right_iris_indices)

            # Use only one eye's iris center to avoid ambiguity (e.g., left iris center)
            gaze_point_x, gaze_point_y = left_iris_center

            if time.time() - calibration_start_time < calibration_duration:
                # During the calibration period, accumulate gaze points to find the neutral position
                calibration_positions.append((gaze_point_x, gaze_point_y))
            else:
                if len(calibration_positions) > 0:
                    # Calculate the average neutral gaze position
                    neutral_gaze_x = np.mean([pos[0] for pos in calibration_positions])
                    neutral_gaze_y = np.mean([pos[1] for pos in calibration_positions])
                    calibration_positions = []  # Clear calibration data after use

                # Calculate the deviation from the neutral position
                delta_x = gaze_point_x - neutral_gaze_x
                delta_y = gaze_point_y - neutral_gaze_y

                # Determine the direction based on the deviation
                if abs(delta_x) > abs(delta_y):  # Horizontal movement
                    if delta_x > 0:
                        direction = 'right'
                    else:
                        direction = 'left'
                else:  # Vertical movement
                    if delta_y > 0:
                        direction = 'down'
                    else:
                        direction = 'up'

                # Check if the gaze has been stable within the threshold area
                if abs(delta_x) < stability_threshold and abs(delta_y) < stability_threshold:
                    # If gaze is stable, update the last stable time
                    last_stable_time = time.time()
                else:
                    # If gaze is not stable and the stability time has been exceeded, move the cursor
                    if time.time() - last_stable_time > stability_time_threshold:
                        if direction == 'right':
                            current_screen_x += move_speed
                        elif direction == 'left':
                            current_screen_x -= move_speed
                        elif direction == 'down':
                            current_screen_y += move_speed
                        elif direction == 'up':
                            current_screen_y -= move_speed

                        # Ensure the cursor position stays within screen bounds
                        current_screen_x = np.clip(current_screen_x, 0, screen_width - 1)
                        current_screen_y = np.clip(current_screen_y, 0, screen_height - 1)

                        # Move the mouse cursor to the new position
                        pyautogui.moveTo(current_screen_x, current_screen_y)

            # Detect wink by measuring the vertical distance between the right eye's upper and lower eyelid landmarks
            upper_lid_y = int(face_landmarks.landmark[right_eye_upper_indices[0]].y * frame_height)
            lower_lid_y = int(face_landmarks.landmark[right_eye_lower_indices[0]].y * frame_height)
            eyelid_distance = abs(upper_lid_y - lower_lid_y)

            if(lower_lid_y - upper_lid_y ) < wink_threshold:
                pyautogui.click()




            # Visualize the iris center point
            cv2.circle(frame, (gaze_point_x, gaze_point_y), 5, (255, 0, 0), -1)

            # Optionally, draw the iris and eyelid landmarks
            for idx in left_iris_indices + right_iris_indices + right_eye_upper_indices + right_eye_lower_indices:
                x = int(face_landmarks.landmark[idx].x * frame_width)
                y = int(face_landmarks.landmark[idx].y * frame_height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    else:
        print("No face detected")

    # Display the frame
    cv2.imshow('Eye Gaze Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and destroy windows
cam.release()
cv2.destroyAllWindows()
