import cv2
import matplotlib.pyplot as plt

import signal_detection
import numpy as np


def remove_background(img):
    hh, ww = img.shape[:2]

    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    thresh = cv2.inRange(img, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    mask = 255 - morph

    cv2.imwrite("Assets/Images/thresh.jpg", thresh)

    return thresh


def crop_image(image, points):
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    croped = image[y:y + h, x:x + w].copy()

    pts = points - points.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst


def detect_screen(frame):
    blue_star = cv2.imread('Assets/Images/blue_star.png', cv2.IMREAD_UNCHANGED)
    green_star = cv2.imread('Assets/Images/green_star.png', cv2.IMREAD_UNCHANGED)
    pink_star = cv2.imread('Assets/Images/pink_star.png', cv2.IMREAD_UNCHANGED)

    # second image

    # resize images
    frame_2 = frame
    pink_bottom_right = pink_star
    template_pink = pink_bottom_right[:, :, 0:3]
    alpha_pink = pink_bottom_right[:, :, 3]
    alpha_pink = cv2.merge([alpha_pink, alpha_pink, alpha_pink])
    correlation_pink = cv2.matchTemplate(frame, template_pink, cv2.TM_CCORR_NORMED, mask=alpha_pink)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(correlation_pink)

    # resize images
    frame_1 = frame
    blue_bottom_left = blue_star
    template_blue = blue_bottom_left[:, :, 0:3]
    alpha_blue = blue_bottom_left[:, :, 3]
    alpha_blue = cv2.merge([alpha_blue, alpha_blue, alpha_blue])
    correlation_blue = cv2.matchTemplate(frame, template_blue, cv2.TM_CCORR_NORMED, mask=alpha_blue)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_blue)

    # third image

    # resize images
    frame_3 = frame
    green_bottom_right = green_star

    template_green = green_bottom_right[:, :, 0:3]
    alpha_green = green_bottom_right[:, :, 3]
    alpha_green = cv2.merge([alpha_green, alpha_green, alpha_green])
    correlation_green = cv2.matchTemplate(frame, template_green, cv2.TM_CCORR_NORMED, mask=alpha_green)
    min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(correlation_green)

    # calculations

    top_left = max_loc
    h, w, _ = blue_bottom_left.shape
    bottom_left_blue = (top_left[0], top_left[1] + h)

    top_left2 = max_loc2
    h2, w2, _ = pink_bottom_right.shape
    bottom_right_pink = (top_left2[0] + w2, top_left2[1] + h2)

    top_left3 = max_loc3
    h3, w3, _ = green_bottom_right.shape
    bottom_right_green = (top_left3[0] + w3, top_left3[1] + h3)

    pts = np.array([[bottom_right_green[0] + 25, bottom_right_pink[1] - 4],
                    [bottom_left_blue[0], bottom_right_pink[1] - 14],
                    [bottom_left_blue[0], bottom_right_green[1]-2],
                    [bottom_right_green[0] + 25, bottom_right_green[1]]])

    return pts, frame_3


def read_video(txt_file_name, videoName):
    count = 0

    video = cv2.VideoCapture(videoName)

    while True:
        plt.margins(0, 0)
        if count == 77 * 5:
            print("video recording is done")
            break

        ret, frame = video.read()

        if count == 0:
            pts, frame_3 = detect_screen(frame)

        cropped_image = crop_image(frame, pts)
        cropped_image_thresh = remove_background(cropped_image)

        cv2.imwrite("Assets/Images/Frames-1/frame%d.jpg" % count, cropped_image_thresh)
        image_name1 = "Assets/Images/Frames-1/frame" + str(count) + ".jpg"

        color_arr = [(255, 255, 255)]

        signal, time, non_signal, non_time = signal_detection.read_img(image_name1, txt_file_name, color_arr)

        j = 0

        for i in range(len(signal)):
            if abs(signal[i] - signal[i - 1]) < 20:
                j += 1

        if j == len(signal):
            plt.margins(0, 100)


        cv2.waitKey(1)

        count += 1

    video.release()
    cv2.destroyAllWindows()
