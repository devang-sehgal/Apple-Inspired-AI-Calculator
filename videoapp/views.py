import time
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image

# Initialize gemini
genai.configure(api_key="dddaaa")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.75, minTrackCon=0.75)
# detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.75, minTrackCon=0.75)

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

def initialize_canvas(frame):
    return np.zeros_like(frame)

def process_hand(hand):
    lmList = hand["lmList"]  # List of 21 landmarks for the hand
    bbox = hand["bbox"]  # Bounding box around the hand (x,y,w,h coordinates)
    center = hand['center']  # Center coordinates of the hand
    handType = hand["type"]  # Type of the hand ("Left" or "Right")
    fingers = detector.fingersUp(hand)  # Count the number of fingers up
    return lmList, bbox, center, handType, fingers

def weighted_average(current, previous, alpha=0.5):
    return alpha * current + (1 - alpha) * previous

response_text = None

mutexLock, wait_time = 0, 100

def send_to_ai(model, canvas, fingers):
    global response_text
    if mutexLock == 0:
        image = Image.fromarray(canvas)
        response = model.generate_content(["The image you see is a math equation written in very bad handwriting, solve it and calculate the answer. ANSWER FORMAT: 1st Line - only answer, and add a line break; next two lines - approach in plain text.", image])
        print("Response Text: ", response_text)
        response_text = response.text if response else None

# Initialize variables
prev_pos = None
drawing = False
points = []  # Store points for drawing
smooth_points = None  # Smoothed position

# Initialize canvas
_, frame = cap.read()
canvas = initialize_canvas(frame)

def video_stream():
    global prev_pos, drawing, points, smooth_points, canvas

    while True:
        # Capture each frame from the webcam
        global mutexLock
        mutexLock = (mutexLock + 1) % wait_time
        success, img = cap.read()

        if not success:
            print("Failed to capture image")
            break

        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, draw=True, flipType=True)

        if hands:
            hand = hands[0]
            lmList, bbox, center, handType, fingers = process_hand(hand)

            # Get the positions of the index and middle finger tips

            # get the second and third landmarks of the index finger
            index_tip = lmList[8]
            index_x, index_y, z = lmList[8]
            thumb_x, thumb_y, z2 = lmList[4]

            # print((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
            # Determine drawing state based on fingers up
            # if fingers[1] == 1 and fingers[2] == 0:  # Only index finger is up

            if (thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2 < 2000:
                current_pos = np.array([index_tip[0], index_tip[1]])
                if smooth_points is None:
                    smooth_points = current_pos
                else:
                    smooth_points = weighted_average(current_pos, smooth_points)
                smoothed_pos = tuple(smooth_points.astype(int))

                if drawing:  # Only add to points if already drawing
                    points.append(smoothed_pos)
                prev_pos = smoothed_pos
                drawing = True
            
            if (thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2 >= 2000:  # Index finger is up
                drawing = False
                prev_pos = None
                points = []  # Clear points to avoid connection
                smooth_points = None

            if (thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2 >= 50000:  # Thumb is up
                canvas = initialize_canvas(img)
                points = []
                drawing = False
                prev_pos = None
                smooth_points = None

            if fingers[4] == 1:
                send_to_ai(model, canvas, fingers)

        # Draw polyline on the canvas
        if len(points) > 1 and drawing:
            cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=5)

        # Combine the image and canvas
        img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_response(request):
    global response_text
    return JsonResponse({'response': response_text})
