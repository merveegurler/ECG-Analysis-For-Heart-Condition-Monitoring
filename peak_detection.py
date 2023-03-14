import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import os
import signal_detection


def get_peaks(sig_peaks, signals):
    peaks = []
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if sig_peaks[i] == 1:
            peaks.append([i / 1000, signals.ECG_Clean[i]])
    peaks = np.array(peaks)
    peaks = np.transpose(peaks)
    return peaks


def plot_ecg(signals):
    valid_index = signals.first_valid_index()
    plt.plot([(x + valid_index) / 1000 for x in range(len(signals))], signals.ECG_Clean, label="Clean signal")


def plot_p_peaks(signals):
    peaks = get_peaks(signals.ECG_P_Peaks, signals)
    plt.scatter(peaks[0], peaks[1], marker="o", color="green", label="P Peaks")


def plot_q_peaks(signals):
    peaks = get_peaks(signals.ECG_Q_Peaks, signals)
    plt.scatter(peaks[0], peaks[1], marker="o", color="cyan", label="Q Peaks")


def plot_r_peaks(signals):
    peaks = get_peaks(signals.ECG_R_Peaks, signals)
    plt.scatter(peaks[0], peaks[1], marker="o", color="red", label="R Peaks")


def plot_period(signals):
    peaks = get_peaks(signals.ECG_R_Peaks, signals)
    for i in range(len(peaks[0])):
        plt.vlines(peaks[0][i], signals.ECG_Clean.min() - 0.1, signals.ECG_Clean.max() + 0.1, color="red")


def plot_numbers(signals):
    peaks = get_peaks(signals.ECG_R_Peaks, signals)
    for i in range(len(peaks[0])):
        plt.text(peaks[0][i], peaks[1][i], str(i + 1))


def plot_s_peaks(signals):
    peaks = get_peaks(signals.ECG_S_Peaks, signals)
    plt.scatter(peaks[0], peaks[1], marker="o", color="orange", label="S Peaks")


def plot_t_peaks(signals):
    peaks = get_peaks(signals.ECG_T_Peaks, signals)
    plt.scatter(peaks[0], peaks[1], marker="o", color="blue", label="T Peaks")


def ecg_from_file(lead, file_path):
    ecg = []
    with open(os.path.abspath(file_path), encoding="utf-8") as file:
        for line in file:
            if line[0] != ";":
                currline = line.split()
                ecg.append(float(currline[lead - 1]) * 0.0001185)
    return ecg


def show_ecg(lead, file_path):
    if file_path != "" and 1 <= lead <= 3:
        ecg = ecg_from_file(lead, file_path)
        signals, info = nk.ecg_process(ecg)
        plot_ecg(signals)
        plot_p_peaks(signals)
        plot_q_peaks(signals)
        plot_r_peaks(signals)
        plot_s_peaks(signals)
        plot_t_peaks(signals)
        plot_period(signals)
        plot_numbers(signals)
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Time [sec]")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()


def koef(signals):
    valid_index = signals.first_valid_index()
    return signals.ECG_Clean[valid_index] * -1


def zeroing(koef, signals):
    signals.ECG_Clean += koef


def get_p_info(signals):
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_P_Peaks[i] == 1:
            result = round(signals.ECG_Clean[i], 2)
            break
    return result


def get_q_info(signals):
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_Q_Peaks[i] == 1:
            result = round(signals.ECG_Clean[i], 2)
            break
    return result


def get_r_info(signals):
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_R_Peaks[i] == 1:
            result = round(signals.ECG_Clean[i], 2)
            break
    return result


def get_s_info(signals):
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_S_Peaks[i] == 1:
            result = round(signals.ECG_Clean[i], 2)
            break
    return result


def get_t_info(signals):
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_T_Peaks[i] == 1:
            result = round(signals.ECG_Clean[i], 2)
            break
    return result


def get_pq_info(signals):
    start = 0
    end = 0
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_P_Peaks[i] == 1:
            start = i
        if signals.ECG_Q_Peaks[i] == 1:
            end = i
            break
    return round((end - start), 2)


def get_st_info(signals):
    start = 0
    end = 0
    valid_index = signals.first_valid_index()
    for i in range(valid_index, len(signals) + valid_index):
        if signals.ECG_S_Peaks[i] == 1:
            start = i
        if signals.ECG_T_Peaks[i] == 1:
            end = i
            break
    return round((end - start), 2)


def get_info(signals):
    info = [
        ["P", get_p_info(signals)],
        ["Q", get_q_info(signals)],
        ["R", get_r_info(signals)],
        ["S", get_s_info(signals)],
        ["T", get_t_info(signals)],
        ["P", "48 ms"],
        ["Q", "9 ms"],
        ["R", "36 ms"],
        ["S", "31 ms"],
        ["T", "180 ms"],
        ["PQ", get_pq_info(signals)],
        ["ST", get_st_info(signals)]
    ]
    return info


def plot_info(signals):
    info = get_info(signals)
    plt.table(cellText=info, bbox=[1, 0, 0.1, 1])


def show_complex(lead, file_path, num):
    if file_path != "" and 1 <= lead <= 3:
        ecg = ecg_from_file(lead, file_path)
        signals, info = nk.ecg_process(ecg)
        if num.isdigit():
            num = int(num)
            if 0 < num <= len(info["ECG_R_Peaks"]):
                peak = info["ECG_R_Peaks"][num - 1]
                if peak - 200 > 0 and peak + 350 < len(signals):
                    signals = signals[peak - 200:peak + 350]
                    koef_ = koef(signals)
                    zeroing(koef_, signals)
                    plot_ecg(signals)
                    plot_p_peaks(signals)
                    plot_q_peaks(signals)
                    plot_r_peaks(signals)
                    plot_s_peaks(signals)
                    plot_t_peaks(signals)
                    plot_info(signals)
                    plt.ylabel("Amplitude [mV]")
                    plt.xlabel("Time [sec]")
                    plt.legend(loc="upper right")
                    plt.grid()
                    plt.show()


def coherent_accumulation_method(lead, file_path):
    if file_path != "" and 1 <= lead <= 3:
        ecg = ecg_from_file(lead, file_path)
        signals, info = nk.ecg_process(ecg)
        hits = []
        for i in range(2, 10):
            peak = info["ECG_R_Peaks"][i]
            hits.append(signals.ECG_Raw[peak - 200:peak + 350])
        hits = np.array(hits)
        hits = hits.transpose()
        to_plot = []
        for el in hits:
            to_plot.append(sum(el) / 8)
        plt.plot([x / 1000 for x in range(len(hits))], to_plot, label="Signal after coherent accumulation")
        plot_info(signals)
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Time [sec]")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()


def write_info(lead, file_path):
    if file_path != "" and 1 <= lead <= 3:
        ecg = ecg_from_file(lead, file_path)
        signals, info = nk.ecg_process(ecg)
        infoo = get_info(signals)

    return infoo


def peak_detection(crv_filenames, csv_filename):
    info1 = write_info(1, crv_filenames[0])
    info2 = write_info(1, crv_filenames[1])

    row = [2, float(info1[0][1]), float(info1[4][1]), float(info1[2][1]),
           float(info1[3][1]), float(info1[1][1]), float(info1[10][1]),
           float(info1[11][1]), float(info2[0][1]),
           float(info2[4][1]), float(info2[2][1]), float(info2[3][1]),
           float(info2[1][1]), float(info2[10][1]), float(info2[11][1])]

    signal_detection.write_to_csv(csv_filename, row)
