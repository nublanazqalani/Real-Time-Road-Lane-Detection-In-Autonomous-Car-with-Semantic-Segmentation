import cv2
import numpy as np

cap = cv2.VideoCapture("tcp://193.168.0.1:6200/")

while True:
    _, frame = cap.read()
    # print(frame.shape)

    # cv2.circle(frame, (155, 120), 5, (0, 0, 255), 3)
    # cv2.circle(frame, (480, 120), 5, (0, 0, 255), 3)
    # cv2.circle(frame, (1, 1500), 5, (0, 0, 255), 3)
    # cv2.circle(frame, (2500, 1400), 5, (0, 0, 255), 3)

    ##### Ukuran 2K (2560x1600)
    # cv2.line(frame, (0, 1400), (2560, 1400), (0, 0, 255), 3)
    # cv2.line(frame, (0, 1400), (1000, 950), (0, 0, 255), 3)
    # cv2.line(frame, (1560, 950), (2560, 1400), (0, 0, 255), 3)
    # cv2.line(frame, (1560, 950), (1000, 950), (0, 0, 255), 3)

    ##### Ukuran realtime (800x480)
    cv2.line(frame, (0, 420), (800, 420), (0, 0, 255), 3)
    cv2.line(frame, (0, 420), (300, 285), (0, 0, 255), 3)
    cv2.line(frame, (468, 285), (800, 420), (0, 0, 255), 3)
    cv2.line(frame, (468, 285), (300, 285), (0, 0, 255), 3)

    #jarak anatar kedua roda 750 pixel (2560x1600)
    cv2.circle(frame, (1630, 1300), 5, (0, 0, 255), 3)
    cv2.circle(frame, (880, 1300), 5, (0, 0, 255), 3)

    # # jarak anatar kedua roda 750 pixel (800x480)
    # cv2.circle(frame, (489, 390), 5, (0, 0, 255), 3)
    # cv2.circle(frame, (240, 390), 5, (0, 0, 255), 3)

    # ##### Warp 2560x1600 to 400x600
    # pts1 = np.float32([[0, 1400], [2560, 1400], [1000, 950], [1560, 950]])
    # pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
    # matrix = cv2.getPerspectiveTransform(pts1,pts2)

    ##### Warp 800x480 to 400x600
    pts1 = np.float32([[0, 420], [800, 420], [300, 285], [468, 285]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result_warp = cv2.warpPerspective(frame, matrix, (400, 600))
    result_flipped = cv2.flip(result_warp, 0)

    cv2.circle(result_flipped, (270, 600), 5, (0, 0, 255), -1)
    cv2.circle(result_flipped, (185, 600), 5, (0, 0, 255), -1)


    cv2.imshow("Frame", frame)
    cv2.imshow("IPM Video",  result_flipped)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

