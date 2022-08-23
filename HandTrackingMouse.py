import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy


####################################
w_cam, h_cam = 640, 480
frame_reduction = 100
shake_reduction = 8
####################################
pTime = 0
previous_loc_x, previous_loc_y = 0,0
current_loc_x, current_loc_y = 0,0

cap = cv2.VideoCapture(1)
cap.set(3, w_cam)
cap.set(4, h_cam)
detector = htm.hand_detector(max_hands=1)
w_screen, h_screen = autopy.screen.size()
# print(w_screen,h_screen)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list, bbox = detector.find_position(img)

    if len(lm_list) != 0:
        x1,y1 = lm_list[8][1:]
        x2,y2 = lm_list[12][1:]
        # print(x1,y1,x2,y2)

        fingers = detector.fingers_up()
        # print(fingers)

        # Index Finger - Moving
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.rectangle(img,(frame_reduction, frame_reduction),(w_cam - frame_reduction,h_cam - frame_reduction),
                          (0,255,200), 2)
            x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, w_screen))
            y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, h_screen))

            # Shake Mouse Reduction
            current_loc_x = previous_loc_x + (x3 - previous_loc_x) / shake_reduction
            current_loc_y = previous_loc_y + (y3 - previous_loc_y) / shake_reduction

            # Move Mouse
            autopy.mouse.move(w_screen - current_loc_x, current_loc_y)
            cv2.circle(img,(x1,y1), 10,(200,0,200),cv2.FILLED)
            previous_loc_x, previous_loc_y = current_loc_x, current_loc_y


        # Index Finger and middle finger are up - click

        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, line_info = detector.find_distance(8, 12, img)
            # print(length)
            # Click mouse if distance is short
            if length < 40:
                cv2.circle(img,(line_info[4],line_info[5]), 10,(173,255,47),cv2.FILLED)
                autopy.mouse.click()

    #####################################
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)