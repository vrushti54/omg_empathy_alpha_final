import numpy as np
import librosa

"""
Librosa-based feature extraction used during preprocessing.

Key point: we expose extract_features(y, sr, hop_samples=...)
so the hop length (frame step) matches your config [stft] hop_size.
That keeps feature frames aligned with the 25 FPS annotations.
"""

def _pow2_ceiling(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def extract_logmel(y, sr, n_mels=64, n_fft=1024, hop_length=640):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    logmel = librosa.power_to_db(S, ref=np.max)
    return logmel.T  # (time, mel)

def extract_mfcc(y, sr, n_mfcc=13, n_fft=1024, hop_length=640):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc.T  # (time, coeff)

def rms_energy(y, frame_length=1024, hop_length=640):
    return librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]  # (time,)

def spectral_centroid(y, sr, n_fft=1024, hop_length=640):
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]  # (time,)

def extract_features(y, sr=16000, hop_samples=640, win_samples=None):
    """
    Concatenate log-mel (64), MFCC (13), RMS (1), Spectral Centroid (1).
    hop_samples MUST match config [stft] hop_size for alignment.

    Returns: np.ndarray of shape (T, 79)
    """
    if win_samples is None:
        # a reasonable analysis window (>= hop); power-of-two FFT size
        win_samples = max(hop_samples * 2, 1024)
    n_fft = _pow2_ceiling(win_samples)

    logmel   = extract_logmel(y, sr, n_fft=n_fft, hop_length=hop_samples)   # (T, 64)
    mfcc     = extract_mfcc(y, sr, n_fft=n_fft, hop_length=hop_samples)     # (T, 13)
    rms      = rms_energy(y, frame_length=n_fft, hop_length=hop_samples)    # (T,)
    centroid = spectral_centroid(y, sr, n_fft=n_fft, hop_length=hop_samples)# (T,)

    # Align lengths (defensive)
    T = min(logmel.shape[0], mfcc.shape[0], len(rms), len(centroid))
    logmel   = logmel[:T]
    mfcc     = mfcc[:T]
    rms      = rms[:T].reshape(-1, 1)
    centroid = centroid[:T].reshape(-1, 1)

    feats = np.concatenate([logmel, mfcc, rms, centroid], axis=1)  # (T, 79)
    print(f"[extract_features] sr={sr}, hop={hop_samples} samples -> features {feats.shape}")
    return feats
