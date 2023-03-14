from threading import Thread
import cv2, time
from matplotlib import pyplot as plt

import signal_detection
import video_processing


class VideoStreamWidget(object):
    def __init__(self, src="https://192.168.1.20:8080/video"):
        self.capture = cv2.VideoCapture(src)
        self.frame = cv2.imread("Assets/Images/normal_frame.jpg")
        self.output_file_name = "Assets/raw_data/deneme_v5.txt"
        self.count = 0
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.thread2 = Thread(target=self.process, args=())
        self.thread2.daemon = True
        self.thread2.start()


    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)


    def process(self):
        while self.capture.isOpened():
            print("geliyor")
            if self.count == 0:
                pts, frame_3 = video_processing.detect_screen(self.frame)

            cropped_image = video_processing.crop_image(self.frame, pts)
            cropped_image_thresh = video_processing.remove_background(cropped_image)

            cv2.imwrite("Assets/Images/Frames-1/frame%d.jpg" % self.count, cropped_image_thresh)
            image_name1 = "Assets/Images/Frames-1/frame" + str(self.count) + ".jpg"

            color_arr = [(255, 255, 255)]
            signal, timee = signal_detection.read_img(image_name1, self.output_file_name, color_arr)

            j = 0
            for i in range(len(signal)):
                if abs(signal[i] - signal[i - 1]) < 20:
                    j += 1

            if j == len(signal):
                plt.margins(0, 100)

            plt.plot(timee, signal)
            plt.savefig("Assets/Images/plot.jpg")
            plt.clf()

            time.sleep(0.01)


    def show_frame(self):
        cv2.imshow('frame', self.frame)
        plot = cv2.imread("Assets/Images/plot.jpg")
        cv2.imshow("plot", plot)
        cv2.waitKey(1)
        """if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)"""


if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
