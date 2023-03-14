import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image

import video_processing

header = ["type", "0_pPeak", "0_tPeak", "0_rPeak", "0_sPeak", "0_qPeak", "0_pq_interval",
          "0_st_interval", "1_pPeak", "1_tPeak", "1_rPeak", "1_sPeak", "1_qPeak",
          "1_pq_interval", "1_st_interval"]


def mse(count):
    imageB = Image.open("Assets/Images/plot.jpg")

    box = (118, 36, 847, 228)

    imageB = imageB.crop(box)

    rgb = imageB.convert('RGB')
    for x in range(imageB.size[0]):
        for y in range(imageB.size[1]):
            r, g, r_b, = rgb.getpixel((x, y))
            if r >= 150 and g >= 150 and r_b >= 150:
                imageB.putpixel((x, -y), (0, 0, 0))
            else:
                imageB.putpixel((x, -y), (255, 255, 255))

    imageB = imageB.resize((940, 300))
    imageB.save("Assets/Images/imageB.png")

    path = "Assets/Images/Frames-1/frame" + str(count) + ".jpg"

    imageA = cv2.imread(path)

    imageA = cv2.resize(imageA, (940, 300), interpolation = cv2.INTER_AREA)
    imageB = cv2.imread("Assets/Images/imageB.png")

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err/100


def write_to_csv(filepath, write_csv):
    with open(filepath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(write_csv)


def write_to_txt(filepath, y_coordinates):
    with open(filepath, 'w') as file:
        for i in range(len(y_coordinates)):
            file.write(str(y_coordinates[i]) + "\n")


def filter_signal(time, ecg_measurements, lowcut, highcut, signal_freq, filter_order):
    """np.random.seed(42)  # for reproducibility
    fs = 30  # sampling rate, Hz
    ts = time  # time vector - 5 seconds
    ys = signal  # signal @ 1.0 Hz, without noise
    yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    yraw = ys + yerr

    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
    y_lfilter = scipy.signal.lfilter(b, a, yraw)

    # apply filter forward and backward using filtfilt
    y_filtfilt = scipy.signal.filtfilt(b, a, yraw)

    plt.plot(ts, yraw, label="Raw signal")
    plt.plot(ts, y_lfilter, alpha=0.8, lw=3, label="SciPy lfilter")
    plt.plot(ts, y_filtfilt, alpha=0.1, lw=2, label="SciPy filtfilt")

    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
               ncol=2, fontsize="smaller")

    plt.show()"""

    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = scipy.signal.butter(filter_order, [low, high], btype="band")
    y = scipy.signal.lfilter(b, a, ecg_measurements)

    """plt.plot(time, ecg_measurements, label="Non-normalized Data")
    plt.show()"""

    return y



def find_rgb(n_img, color_arr):
    x_coordinates = []
    y_coordinates = []

    rgb = n_img.convert('RGB')
    for x in range(n_img.size[0]):
        for y in range(n_img.size[1]):
            r, g, r_b, = rgb.getpixel((x, y))
            for i in range(len(color_arr)):
                r_query, g_query, b_query = color_arr[i]
                if r == r_query and g == g_query and r_b == b_query:
                    x_coordinates.append(x)
                    y_coordinates.append(-y)
    return x_coordinates, y_coordinates


def normalize_signal(ecg_measurements, time):
    filtered_ecg = []
    filtered_time = []

    """for i in range(len(ecg_measurements)):
        if i > 0:
            if abs(ecg_measurements[i] - ecg_measurements[i - 1]) < 10:
                filtered_ecg.append(filtered_ecg[i-1])
            else:
                filtered_ecg.append(ecg_measurements[i])
        else:
            filtered_ecg.append(ecg_measurements[i])"""

    for i in range(len(time)):
        if i > 0:
            if abs(time[i] - time[i - 1]) < 0.3:
                filtered_time.append(filtered_time[i - 1])
            else:
                filtered_time.append(time[i])
        else:
            filtered_time.append(time[i])

    """print(filtered_time)
    print(filtered_ecg)"""

    return ecg_measurements, filtered_time


def read_img(imagefile, filename_txt, colors):
    n_img = Image.open(imagefile)
    w, h = n_img.size

    box = (15, 0, w, h)
    n_img = n_img.crop(box)

    x_coordinates, y_coordinates = find_rgb(n_img, colors)
    """filter_signal(x_coordinates, y_coordinates, 0.7, 7.0, 70, 2)"""
    y, x = normalize_signal(y_coordinates, x_coordinates)

    write_to_txt(filename_txt, y)

    return y, x
