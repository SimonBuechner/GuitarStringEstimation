import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks, windows
from scipy.fftpack import fft
from sympy.strategies.core import switch


def parainterp(idx, data):
    if idx <= 0 or idx >= len(data) - 1:
        return idx, 0.0, data[idx]

    xy = data[idx - 1:idx + 2]
    p = 1  # Das Maximum ist definitionsgemäß in der Mitte (idx=1)

    # Berechnung der parabolischen Verschiebung
    delta = 0.5 * (xy[p - 1] - xy[p + 1]) / (xy[p - 1] - 2 * xy[p] + xy[p + 1])
    newMax = xy[p] - 0.25 * (xy[p - 1] - xy[p + 1]) * delta

    return idx + delta, delta, newMax


def estimate_frequency_from_phase(prev_phase, current_phase, k, W, H, sr):
    omega_k = 2 * np.pi * k / W
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi, 2 * np.pi) - np.pi
    return delta_phi / (2 * np.pi * H) * sr


fs = 16000  # Sampling rate
freq_range = np.arange(82, 360, 1)  # Frequenzen von 90 bis 130 Hz
L = 3  # Signal-Länge in Sekunden
t = np.arange(0, L, 1 / fs)  # Zeitvektor
N_array = [512, 1024, 2048, 4096, int(0.1 * fs), int(0.3 * fs)]

fmin, fmax = np.min(freq_range), np.max(freq_range)  # Grenzwerte für Yin & pYin

errors_fft = np.zeros((len(freq_range), len(N_array)))
errors_para_interp = np.zeros((len(freq_range), len(N_array)))
errors_instphase = np.zeros((len(freq_range), len(N_array)))
errors_yin = np.zeros((len(freq_range), len(N_array)))
errors_pyin = np.zeros((len(freq_range), len(N_array)))
betas_fft = np.zeros((len(freq_range), len(N_array)))
betas_para_interp = np.zeros((len(freq_range), len(N_array)))
betas_instphase = np.zeros((len(freq_range), len(N_array)))
betas_yin = np.zeros((len(freq_range), len(N_array)))
betas_pyin = np.zeros((len(freq_range), len(N_array)))

for f_idx, f in enumerate(freq_range):
    sig = np.sin(2 * np.pi * f * t)  # Sinus-Signal

    for i, N in enumerate(N_array):
        R = N // 8  # Hop size
        w = windows.hann(N, sym=True)  # Hann-Fenster

        num_frames = len(sig) // R - 1
        errors_local = []

        # Initialisiere das erste Segment für prev_segment
        start = 0
        end = start + N
        if end > len(sig):
            continue  # Falls das Signal zu kurz ist, überspringen

        prev_segment = sig[start:end] * w

        # Yin & pYin berechnen
        hop_length = N // 16  # Ähnlich wie R
        yin_freqs = librosa.yin(sig, fmin=fmin, fmax=fmax, sr=fs, frame_length=N, hop_length=hop_length)
        pyin_freqs = librosa.pyin(sig, fmin=fmin, fmax=fmax, sr=fs, frame_length=N, hop_length=hop_length)[0]

        for frame in range(1, num_frames):  # Start ab Frame 1 (Index 2)
            start = frame * R
            end = start + N
            if end > len(sig):
                break

            segment = sig[start:end] * w
            SIG = np.abs(fft(segment, N))[:N // 2 + 1]
            SIG_log = 20 * np.log10(SIG + 1e-10)  # Logarithmische Darstellung mit Schutz gegen log(0)

            peaks, _ = find_peaks(SIG_log, height=-30)
            if len(peaks) == 0:
                continue

            pos = peaks[0]
            freq_fft = pos * fs / N

            p, delta, _ = parainterp(pos, SIG_log)
            est_freq_paraInterp = (p) * fs / N  # Korrigierte Skalierung

            prev_phase = np.angle(fft(prev_segment, N))[pos]
            current_phase = np.angle(fft(segment, N))[pos]
            instphase_freq = estimate_frequency_from_phase(prev_phase, current_phase, pos, N, R, fs)

            prev_segment = segment  # Aktualisiere prev_segment

            yin_est = yin_freqs[frame] if frame < len(yin_freqs) and not np.isnan(yin_freqs[frame]) else f
            pyin_est = pyin_freqs[frame] if frame < len(pyin_freqs) and not np.isnan(pyin_freqs[frame]) else f

            # Berechne Beta für jede Methode
            beta_fft = (freq_fft / f) ** 2 - 1
            beta_paraInterp = (est_freq_paraInterp / f) ** 2 - 1
            beta_instphase = (instphase_freq / f) ** 2 - 1
            beta_yin = (yin_est / f) ** 2 - 1
            beta_pyin = (pyin_est / f) ** 2 - 1

            # Fehler berechnen
            errors_local.append(
                [abs(f - freq_fft), abs(f - est_freq_paraInterp), abs(f - instphase_freq), abs(f - yin_est),
                 abs(f - pyin_est)])

            # Speichere die berechneten Betas
            betas_fft[f_idx, i] += beta_fft
            betas_para_interp[f_idx, i] += beta_paraInterp
            betas_instphase[f_idx, i] += beta_instphase
            betas_yin[f_idx, i] += beta_yin
            betas_pyin[f_idx, i] += beta_pyin

        if errors_local:
            errors_fft[f_idx, i], errors_para_interp[f_idx, i], errors_instphase[f_idx, i], errors_yin[f_idx, i], \
            errors_pyin[f_idx, i] = \
                np.mean(errors_local, axis=0)

# Ausgabe der Betas und Frequenzabweichungen für N = 4096
print("\nInstantaneous Phase: Betas und Frequenzabweichungen für N = 4096:")
for f_idx, f in enumerate(freq_range):
    print(f"\nFrequenz: {f} Hz:  Beta = {betas_instphase[f_idx, 3]:.8f}, Frequenzabweichung = {errors_instphase[f_idx, 3]:.8f} Hz")



# Console output for mean errors and beta values per window size (N)
print("Mean Errors and Beta per Window Size (N):")
for i, N in enumerate(N_array):
    print(f"\nWindow Size (N) = {N}:")

    # FFT
    print(
        f"  FFT: {np.mean(errors_fft[:, i]):.8f} Hz, Max: {np.max(errors_fft[:, i]):.8f}, Var: {np.var(errors_fft[:, i]):.8f}, Std: {np.std(errors_fft[:, i]):.8f}, Beta: {np.mean(betas_fft[:, i]):.8f}, Beta Max: {np.max(betas_fft[:, i]):.8f}, Beta Var: {np.var(betas_fft[:, i]):.8f}, Beta Std: {np.std(betas_fft[:, i]):.8f}")
    # Parabolic Interpolation
    print(
        f"  Parabolic Interpolation: {np.mean(errors_para_interp[:, i]):.8f} Hz, Max: {np.max(errors_para_interp[:, i]):.8f}, Var: {np.var(errors_para_interp[:, i]):.8f}, Std: {np.std(errors_para_interp[:, i]):.8f}, Beta: {np.mean(betas_para_interp[:, i]):.8f}, Beta Max: {np.max(betas_para_interp[:, i]):.8f}, Beta Var: {np.var(betas_para_interp[:, i]):.8f}, Beta Std: {np.std(betas_para_interp[:, i]):.8f}")
    # Instantaneous Phase
    print(
        f"  Instantaneous Phase: {np.mean(errors_instphase[:, i]):.8f} Hz, Max: {np.max(errors_instphase[:, i]):.8f}, Var: {np.var(errors_instphase[:, i]):.8f}, Std: {np.std(errors_instphase[:, i]):.8f}, Beta: {np.mean(betas_instphase[:, i]):.8f}, Beta Max: {np.max(betas_instphase[:, i]):.8f}, Beta Var: {np.var(betas_instphase[:, i]):.8f}, Beta Std: {np.std(betas_instphase[:, i]):.8f}")
    # Yin
    print(
        f"  Yin: {np.mean(errors_yin[:, i]):.8f} Hz, Max: {np.max(errors_yin[:, i]):.8f}, Var: {np.var(errors_yin[:, i]):.8f}, Std: {np.std(errors_yin[:, i]):.8f}, Beta: {np.mean(betas_yin[:, i]):.8f}, Beta Max: {np.max(betas_yin[:, i]):.8f}, Beta Var: {np.var(betas_yin[:, i]):.8f}, Beta Std: {np.std(betas_yin[:, i]):.8f}")
    # pYin
    print(
        f"  pYin: {np.mean(errors_pyin[:, i]):.8f} Hz, Max: {np.max(errors_pyin[:, i]):.8f}, Var: {np.var(errors_pyin[:, i]):.8f}, Std: {np.std(errors_pyin[:, i]):.8f}, Beta: {np.mean(betas_pyin[:, i]):.8f}, Beta Max: {np.max(betas_pyin[:, i]):.8f}, Beta Var: {np.var(betas_pyin[:, i]):.8f}, Beta Std: {np.std(betas_pyin[:, i]):.8f}")