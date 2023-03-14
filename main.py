from matplotlib import pyplot as plt
from PIL import Image
import disease_detection
import signal_detection
import video_processing
import peak_detection
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2


class Application(QMainWindow):
    def __init__(self):
        super(Application, self).__init__()
        self.frame = None
        self.finish = 0
        self.count = 0


        self.setGeometry(50, 50, 1040, 1000)
        self.setStyleSheet("background-color: rgb(12, 45, 72);")
        self.setWindowTitle("ECG ANALYSIS FOR HEART CONDITION MONITORING")
        self.setWindowIcon(QtGui.QIcon('Assets/Images/icon.png'))

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout_image = QtWidgets.QHBoxLayout(self)
        self.layout_plot = QtWidgets.QHBoxLayout(self)

        self.start = QtWidgets.QLabel(self)
        self.l2_button = QtWidgets.QPushButton(self)
        self.v5_button = QtWidgets.QPushButton(self)
        self.l2_button_2 = QtWidgets.QPushButton(self)
        self.v5_button_2 = QtWidgets.QPushButton(self)
        self.result_button = QtWidgets.QPushButton(self)
        self.read_data = QtWidgets.QLabel(self)
        self.incoming_data = QtWidgets.QLabel(self)
        self.initUI()
        self.l2_text = QtWidgets.QLabel(self)
        self.l2_text_2 = QtWidgets.QLabel(self)
        self.v5_text = QtWidgets.QLabel(self)
        self.v5_text_2 = QtWidgets.QLabel(self)
        self.prediction = QtWidgets.QLabel(self)
        self.output_file_name = ""
        self.label = QtWidgets.QLabel(self)
        self.label2 = QtWidgets.QLabel(self)
        self.result_label = QtWidgets.QLabel(self)

        self.done = QtWidgets.QInputDialog(self)
        self.image_name2 = "Assets/Images/normal_frame.jpg"

        self.layout.addWidget(self.start)
        self.layout.addWidget(self.l2_button)
        self.layout.addWidget(self.v5_button)
        self.layout.addWidget(self.l2_button_2)
        self.layout.addWidget(self.v5_button_2)
        self.layout.addWidget(self.read_data)
        self.layout.addWidget(self.incoming_data)
        self.layout.addWidget(self.l2_text)
        self.layout.addWidget(self.l2_text_2)
        self.layout.addWidget(self.v5_text)
        self.layout.addWidget(self.v5_text_2)
        self.layout.addWidget(self.prediction)
        self.layout.addWidget(self.result_button)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.done)
        self.layout_image.addWidget(self.label)
        self.layout_plot.addWidget(self.label2)


    def read_video(self):

        count = 0

        video = cv2.VideoCapture(self.videoName)

        while True:
            plt.margins(0, 0)
            if self.finish:
                print("video recording is done")
                break

            ret, frame = video.read()

            if count == 0:
                pts, frame_3 = video_processing.detect_screen(frame)

            cropped_image = video_processing.crop_image(frame, pts)
            cropped_image_thresh = video_processing.remove_background(cropped_image)

            cv2.imwrite("Assets/Images/Frames-1/frame%d.jpg" % count, cropped_image_thresh)
            image_name1 = "Assets/Images/Frames-1/frame" + str(count) + ".jpg"

            cv2.imwrite("Assets/Images/normal_frame.jpg", cropped_image)
            image_name2 = "Assets/Images/normal_frame.jpg"

            image2 = Image.open(image_name2)
            image2 = image2.resize((940, 300))
            image2.save("Assets/Images/normal_frame.jpg")

            self.pixmap = QtGui.QPixmap(image_name2)
            self.label.setPixmap(self.pixmap)
            self.label.resize(940, 300)
            self.label.move(50, 50)
            self.show()

            QtWidgets.QApplication.processEvents()

            color_arr = [(255, 255, 255)]
            signal, time = signal_detection.read_img(image_name1, self.output_file_name, color_arr)

            j = 0
            for i in range(len(signal)):
                if abs(signal[i] - signal[i - 1]) < 20:
                    j += 1

            if j == len(signal):
                plt.margins(0, 100)

            plt.plot(time, signal)
            plt.savefig("Assets/Images/plot.jpg")
            plt.clf()

            plot = Image.open("Assets/Images/plot.jpg")
            plot = plot.resize((940, 300))
            plot.save("Assets/Images/plot.jpg")


            self.plot = QtGui.QPixmap("Assets/Images/plot.jpg")
            self.label2.setPixmap(self.plot)
            self.label2.resize(940, 300)
            self.label2.move(50, 400)
            self.show()


            QtWidgets.QApplication.processEvents()

            count += 1

        video.release()
        cv2.destroyAllWindows()


    def initUI(self):
        self.videoName, self.done = QtWidgets.QInputDialog.getText(self, 'Input Dialog', "Enter the video path: ")

        self.incoming_data.setText("Incoming Data: ")
        self.incoming_data.setStyleSheet("color: rgb(255, 255, 255);")
        self.incoming_data.adjustSize()
        self.incoming_data.move(50, 30)

        self.read_data.setText("Read Data: ")
        self.read_data.setStyleSheet("color: rgb(255, 255, 255);")
        self.read_data.adjustSize()
        self.read_data.move(50, 375)

        self.start.setText("Click Lead 2 button to start recording!")
        self.start.setStyleSheet("color: rgb(255, 255, 255);")
        self.start.adjustSize()
        self.start.move(50, 775)

        self.l2_button.setText("Click to start Lead 2 recording")
        self.l2_button.adjustSize()
        self.l2_button.resize(445, 25)
        self.l2_button.move(50, 800)
        self.l2_button.setStyleSheet("border: 0px;\n"
                                     "background-color: rgb(65, 114, 159);\n"
                                     "border-radius:10px;\n" 
                                     "color: white")
        self.l2_button.clicked.connect(self.l2)

        self.v5_button.setText("Click to start Lead V5 recording")
        self.v5_button.adjustSize()
        self.v5_button.resize(445, 25)
        self.v5_button.move(545, 800)
        self.v5_button.setStyleSheet("border: 0px;\n" 
                                     "background-color: rgb(65, 114, 159);\n" 
                                     "border-radius:10px;\n"
                                     "color: white")
        self.v5_button.clicked.connect(self.v5)

        self.result_button.setText("Click to see result!")
        self.result_button.adjustSize()
        self.result_button.resize(450, 25)
        self.result_button.move(295, 835)
        self.result_button.setStyleSheet("border: 0px;\n" 
                                         "background-color: rgb(65, 114, 159);\n" 
                                         "border-radius:10px;\n"
                                         "color: white")
        self.result_button.clicked.connect(self.predict)


    def l2(self):
        QtWidgets.QApplication.processEvents()
        self.l2_button_2.setText("Click to finish Lead 2 recording")
        self.l2_button_2.adjustSize()
        self.l2_button_2.resize(445, 25)
        self.l2_button_2.move(50, 800)
        self.l2_button_2.setStyleSheet("border: 0px;\n"
                                       "background-color: rgb(65, 114, 159);\n"
                                       "border-radius:10px;\n"
                                       "color: white")
        self.l2_button_2.clicked.connect(self.finish_loop)

        QtWidgets.QApplication.processEvents()

        self.l2_text.setText("Lead 2 recording is started! Wait for proccesing...")
        self.l2_text.setStyleSheet("color: rgb(255, 255, 255);")
        self.l2_text.adjustSize()
        self.l2_text.move(50, 870)
        self.l2_text_2.setText("")
        self.l2_text_2.adjustSize()
        QtWidgets.QApplication.processEvents()
        self.l2_text.show()

        self.output_file_name = "Assets/raw_data/deneme_l2.txt"

        self.read_video()

        QtWidgets.QApplication.processEvents()

        self.finish = 0

        QtWidgets.QApplication.processEvents()

        self.l2_text_2.setText("Lead 2 recording is done! Click Lead V5 button for recording!")
        self.l2_text_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.l2_text_2.adjustSize()
        self.l2_text_2.move(50, 895)
        self.l2_text_2.show()


    def v5(self):
        QtWidgets.QApplication.processEvents()
        self.v5_button_2.setText("Click to finish Lead V5 recording")
        self.v5_button_2.adjustSize()
        self.v5_button_2.resize(445, 25)
        self.v5_button_2.move(545, 800)
        self.v5_button_2.setStyleSheet("border: 0px;\n"
                                       "background-color: rgb(65, 114, 159);\n"
                                       "border-radius:10px;\n"
                                       "color: white")
        self.v5_button_2.clicked.connect(self.finish_loop)

        QtWidgets.QApplication.processEvents()

        self.l2_text.setText("Lead V5 recording is started! Wait for proccesing...")
        self.l2_text.adjustSize()
        self.l2_text_2.setText("")
        self.l2_text_2.adjustSize()
        QtWidgets.QApplication.processEvents()

        self.output_file_name = "Assets/raw_data/deneme_v5.txt"

        self.read_video()
        QtWidgets.QApplication.processEvents()

        self.finish = 0

        QtWidgets.QApplication.processEvents()

        peak_detection.peak_detection(["Assets/raw_data/deneme_l2.txt", "Assets/raw_data/deneme_v5.txt"],
                                      "Assets/csv_data/patient.csv")

        self.l2_text_2.setText("Lead V5 recording is done!")
        self.l2_text_2.adjustSize()
        QtWidgets.QApplication.processEvents()


    def finish_loop(self):
        self.finish = 1


    def predict(self):
        self.result_label.setText("Predicted result: ")
        self.result_label.setStyleSheet("color: rgb(255, 255, 255);")
        self.result_label.move(50, 920)
        QtWidgets.QApplication.processEvents()

        prediction = disease_detection.detect_disease("Assets/csv_data/patient.csv")

        print(prediction)
        QtWidgets.QApplication.processEvents()

        self.prediction.setText("%s" % prediction)
        self.prediction.setStyleSheet("color: rgb(255, 255, 255);")
        self.prediction.move(300, 920)
        QtWidgets.QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Application()

    win.show()
    sys.exit(app.exec_())
