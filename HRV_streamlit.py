import streamlit as st
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Streamlitアプリケーションのタイトルを設定
st.title('ECG Analysis App')

# ユーザーが設定できる各パラメータのスライダーを追加
# パラメータ設定
duration = st.sidebar.slider("Duration", min_value=100, max_value=2000, value=900, step=100)
sampling_rate = st.sidebar.slider("Sampling Rate", min_value=200, max_value=1000, value=200, step=100)
heart_rate = st.sidebar.slider("Heart Rate", min_value=40, max_value=150, value=70, step=10)
heart_rate_std = st.sidebar.slider("Heart Rate Standard Deviation", min_value=0, max_value=10, value=2, step=1)
noise = st.sidebar.slider("Noise", min_value=0.01, max_value=5.0, value=0.1, step=0.01)
Noise_range = st.sidebar.slider("Noise Range", min_value=1, max_value=20, value=10, step=1)
Noise_max = st.sidebar.slider("Noise Max", min_value=1, max_value=30, value=24, step=1)


# シミュレートされた心電図の生成
simulated_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate, heart_rate_std=heart_rate_std, noise=noise, method='"multileads"')
ecg_df = pd.DataFrame(simulated_ecg, columns=["ECG"])

# ECG信号の前処理とピーク検出を行う
signals, info = nk.ecg_process(ecg_df["ECG"], sampling_rate=sampling_rate)

# ピークのインデックスを取得
r_peak_indices_ms = (info['ECG_R_Peaks'] / 1) * 1

# ピーク間の時間差を計算
peak_diff = np.diff(r_peak_indices_ms)

# 変更後のピーク間の時間差を計算
modified_peak_diff = peak_diff + np.where(np.random.rand(len(peak_diff)) < 0.8,
                                           np.random.randint(-Noise_range, Noise_range, size=len(peak_diff)),  # 絶対値Noise_range以内の値を生成
                                           np.random.choice(np.concatenate((np.arange(-Noise_max, -10), np.arange(10, Noise_max))), size=len(peak_diff)))

# FFTを計算する
sampling_rate_peak_diff = len(peak_diff) / duration
frequency_range = np.linspace(0, 1, len(peak_diff)) * sampling_rate_peak_diff
fft_peak_diff = np.fft.fft(peak_diff)
fft_modified_peak_diff = np.fft.fft(modified_peak_diff)

# 周波数の範囲を設定する
desired_range = (0.01, 0.6)
start_index = int(desired_range[0] * len(frequency_range))
end_index = int(desired_range[1] * len(frequency_range))
frequency_range = frequency_range[start_index:end_index]
fft_peak_diff = np.abs(fft_peak_diff[start_index:end_index])
fft_modified_peak_diff = np.abs(fft_modified_peak_diff[start_index:end_index])

# プロットを作成する（ピーク間の時間差）
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(frequency_range, fft_peak_diff, label='Standard ECG', color='blue')
ax1.plot(frequency_range, fft_modified_peak_diff, label='Estimated ECG', color='red')
ax1.set_title('ECG Signal FFT')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True)

# LF/HF比率を計算する
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.4)

lf_power = np.sum(fft_peak_diff[int(lf_band[0] * len(frequency_range)):int(lf_band[1] * len(frequency_range))])
hf_power = np.sum(fft_peak_diff[int(hf_band[0] * len(frequency_range)):int(hf_band[1] * len(frequency_range))])
lf_hf_ratio = lf_power / hf_power

lf_power_modified = np.sum(fft_modified_peak_diff[int(lf_band[0] * len(frequency_range)):int(lf_band[1] * len(frequency_range))])
hf_power_modified = np.sum(fft_modified_peak_diff[int(hf_band[0] * len(frequency_range)):int(hf_band[1] * len(frequency_range))])
lf_hf_ratio_modified = lf_power_modified / hf_power_modified

# プロットを作成する（LF/HF比率）
fig2, ax2 = plt.subplots(figsize=(6, 6))
bar_colors = ['blue', 'red']
ax2.bar(['Standard', 'Estimated'], [lf_hf_ratio, lf_hf_ratio_modified], color=bar_colors, alpha=0.5, label='LF/HF Ratio')
ax2.set_title('LF/HF Ratio')
ax2.set_ylabel('Ratio')
ax2.legend()
ax2.grid(True)

# プロットを作成する（ピーク間の時間差のデータ）
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(peak_diff, label='Standard ECG', color='blue')
ax3.plot(modified_peak_diff, label='Estimated ECG', color='red')
ax3.set_title('Peak Interval Data')
ax3.set_xlabel('Sample')
ax3.set_ylabel('Peak Interval')
ax3.legend()
ax3.grid(True)

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(np.arange(len(ecg_df["ECG"]))[:1500] / 200, ecg_df["ECG"][:1500], label='ECG', color='green')
ax4.set_title('ECG Signal')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude')
ax4.legend()
ax4.grid(True)

# グラフを表示
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)
st.pyplot(fig4)
