import cv2
import mediapipe

# Initialize VideoCapture
import pyautogui

cam = cv2.VideoCapture(0)

# Initialize FaceMesh
face_mesh = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    # Read frame from webcam
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
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
            # Iterate through each landmark in face_landmarks.landmark
            for id, landmark in enumerate(face_landmarks.landmark [474:478]):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                  # drawing the circle on the eye.

                cv2.circle(frame, (x, y), 2, (100, 150, 170))

                if id == 1:
                    pyautogui.moveTo(x, y)

    else:
        print("No face detected")

    # Display the frame
    cv2.imshow('Face Landmarks Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and destroy windows
cam.release()
cv2.destroyAllWindows()
