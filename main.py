import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Create a blank canvas (black background)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Drawing variables
prev_x, prev_y = None, None
brush_size = 5
color = (255, 255, 255)  # Default: White
drawing_enabled = True
eraser_enabled = False
drawing_points = []  # Stores points to detect shapes

# Color choices
colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255)
}
color_names = list(colors.keys())
current_color_index = 3  # Start with white

def is_fist_closed(hand_landmarks):
    """Detect if the hand is in a fist (all fingers curled)."""
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    pips = [6, 10, 14, 18]  # PIP joints of the fingers

    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            return False
    return True

def is_thumbs_up(hand_landmarks):
    """Detect if the hand is showing a thumbs-up gesture (for eraser mode)."""
    thumb_tip = hand_landmarks.landmark[4].y
    thumb_mcp = hand_landmarks.landmark[2].y
    index_tip = hand_landmarks.landmark[8].y

    return thumb_tip < thumb_mcp and index_tip > thumb_mcp

def detect_shape(points):
    """Detects if the drawn shape is a circle, square, or triangle."""
    if len(points) < 10:
        return None

    # Convert points to contour format
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) > 8:
        return "circle"
    if len(approx) == 4:
        return "square"
    if len(approx) == 3:
        return "triangle"

    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    frame_height, frame_width, _ = frame.shape

    if canvas.shape[:2] != (frame_height, frame_width):
        canvas = cv2.resize(canvas, (frame_width, frame_height))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * frame_width), int(index_finger.y * frame_height)

            # Check gestures
            drawing_enabled = not is_fist_closed(hand_landmarks)
            eraser_enabled = is_thumbs_up(hand_landmarks)

            # Change color by touching thumb & index finger
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            if abs(thumb_tip.x - index_tip.x) < 0.02 and abs(thumb_tip.y - index_tip.y) < 0.02:
                current_color_index = (current_color_index + 1) % len(color_names)
                color = colors[color_names[current_color_index]]

            if drawing_enabled:
                drawing_points.append((x, y))
                if prev_x is not None and prev_y is not None:
                    if eraser_enabled:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 30)  # Erase
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), color, brush_size)

            prev_x, prev_y = x, y
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None

        # Detect if a shape was drawn
        shape = detect_shape(drawing_points)
        if shape:
            x_min, y_min = np.min(drawing_points, axis=0)
            x_max, y_max = np.max(drawing_points, axis=0)

            if shape == "circle":
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                radius = max((x_max - x_min) // 2, (y_max - y_min) // 2)
                cv2.circle(canvas, (center_x, center_y), radius, color, brush_size)
            elif shape == "square":
                cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, brush_size)
            elif shape == "triangle":
                points = np.array([(x_min, y_max), ((x_min + x_max) // 2, y_min), (x_max, y_max)], np.int32)
                cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=brush_size)

        drawing_points = []  # Clear points after detecting a shape

    # Display brush color
    cv2.rectangle(frame, (10, 10), (60, 60), color, -1)
    cv2.putText(frame, "Color", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    combined_display = np.hstack((frame, canvas))

    cv2.imshow("Webcam & Drawing Canvas", combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(canvas)  # Clear canvas
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
