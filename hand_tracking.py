import cv2
import mediapipe as mp

# seting mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#status object
light_on = False
fan_speed = 0
curtain_open = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    #picture reverse 
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # cvt RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_positions = [] #hand position

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #check index finger higher than new kang
            index_tip = hand_landmarks.landmark[8].y  
            index_mcp = hand_landmarks.landmark[6].y  

            if index_tip < index_mcp:
                light_on = True
            else:
                light_on = False

            #Fan part
            is_pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            is_ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
            is_middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

            if is_pinky_up and not is_ring_up and not is_middle_up:
                fan_speed = 1
            elif is_pinky_up and is_ring_up and not is_middle_up:
                fan_speed = 2
            elif is_pinky_up and is_ring_up and is_middle_up:
                fan_speed = 3
            else:
                fan_speed = 0

            hand_positions.append(hand_landmarks.landmark[0].x * w)
            #curtain
        if len(hand_positions) == 2:
            hand_distance = abs(hand_positions[0] - hand_positions[1])
            if hand_distance > w * 0.5:
                curtain_open = True
            elif hand_distance < w * 0.2:
                curtain_open = False

    #status show
    cv2.putText(frame, f'Light: {"ON" if light_on else "OFF"}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if light_on else (0, 0, 255), 2)
    cv2.putText(frame, f'Fan Speed: {fan_speed}', (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Curtain: {"OPEN" if curtain_open else "CLOSED"}', (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0) if curtain_open else (0, 128, 255), 2)

    cv2.imshow("Hand Tracking Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
