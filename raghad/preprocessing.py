import os
import numpy as np
from scipy.signal import resample , savgol_filter
from scipy.interpolate import interp1d

#===========================================
# STEP1 -  Remove noises using Savgol
#===========================================

def remove_noise(recording): 

    """
    recording: (T, 11) — raw sensor data
    returns:   (T, 11) — smoothed sensor data

    Savitzky-Golay filter smooths noise while preserving sharp movements.
    Sample rate is 50Hz — window of 5 frames = 0.1 seconds.
    Human finger movements take at least 0.1-0.2 seconds to complete.
    polyorder=2 fits a degree-2 curve — not too aggressive.
    """
    if recording.shape[0] < 5:
        return recording 
    return savgol_filter(recording, window_length=5, polyorder=2, axis=0) #returns:(T, 11) — smoothed sensor data

#==========================================
# STEP 2 - Resting Pose Subtraction  
#==========================================

def get_resting_pose(recording, n_resting_rows=120):
    """
    recording:       (T, 11) — raw sensor data
    n_resting_rows:  number of leading rows that represent the resting pose
    returns:         (11,)   — mean over all resting rows as the baseline
    """
    n = min(n_resting_rows, recording.shape[0])
    return recording[0:n, :].mean(axis=0)   # (11,)

def subtract_resting_pose(recording, resting_pose):
    """
    recording:    (T, 11) — raw sensor data
    resting_pose: (11,)   — baseline from get_resting_pose()
    returns:      (T, 11) — every feature now means "how far from rest"
    
    Fixes: uniform glove offset, sensor calibration differences.
    After this step, a value of 0 means "at resting position".
    """
    return recording - resting_pose # (T, 11)

#=================================
# STEP 3 - Relational Features (Hand Size + Body Direction)
#=================================
def compute_relational_features(recording):
    """
    recording: (T, 11) — sensor data after resting pose subtraction
    returns:   (T, 8)  — relational features

    Input columns:
        Col 0  = Thumb
        Col 1  = Index
        Col 2  = Middle
        Col 3  = Ring
        Col 4  = Pinky
        Col 5  = Right Yaw
        Col 6  = Right Pitch
        Col 7  = Right Roll
        Col 8  = Left Yaw
        Col 9  = Left Pitch
        Col 10 = Left Roll

    Output columns:
        Col 0  = Index  - Thumb       (finger spread)
        Col 1  = Middle - Index       (finger spread)
        Col 2  = Ring   - Middle      (finger spread)
        Col 3  = Pinky  - Ring        (finger spread)
        Col 4  = Hand Openness        (mean of all 5 fingers)
        Col 5  = Right Yaw   - Left Yaw
        Col 6  = Right Pitch - Left Pitch
        Col 7  = Right Roll  - Left Roll
    """

    # Inter-finger differences — fixes hand size
    # np.diff computes col[i+1] - col[i] for each neighboring pair
    finger_diffs = np.diff(recording[:, 0:5], axis=1)               # (T, 4)

    # Hand openness — mean of all 5 fingers
    # High value = close hand, Low value = open fist
    openness = recording[:, 0:5].mean(axis=1, keepdims=True)         # (T, 1)

    # Relative IMU — fixes body direction
    # Subtracting left from right removes absolute body orientation
    imu_relative = recording[:, 5:8] - recording[:, 8:11]            # (T, 3)
    return np.concatenate([finger_diffs, openness, imu_relative], axis=1)  # (T, 8)

#=======================================
# STEP 4 - Fixed Length Resampling (Sign Speed)
#=======================================

def normalize_length(recording, target_length=100):
    """
    recording:     (T, 8)           
    target_length: int              — desired output length
    returns:       (target_length, 8) — resampled to fixed length

    Fixes sign speed variation — fast and slow signers
    produce the same length sequence for the LSTM.
    Uses scipy resample which applies anti-aliasing automatically.
    """
    return resample(recording, target_length, axis=0)

#=====================================
#STEP 5 - Delta Features (Motion)
#=====================================

def compute_delta(recording):
    """
    recording: (T, 8) — relational features after resampling
    returns:   (T, 16) — original + delta concatenated

    Delta = difference between consecutive timesteps.
    Captures how fast and in what direction each feature is changing.

    First row delta is set to zeros — no previous frame exists at t=0.
    """

    delta = np.diff(recording, axis=0)                    # (T-1, 8)
    delta = np.vstack([np.zeros((1, 8)), delta])          # (T, 8)  pad first row
    return np.concatenate([recording, delta], axis=1)     # (T, 16)

# ══════════════════════════════════════════════════════════════
# STEP 6 — Augmentation (Training Only)
# ══════════════════════════════════════════════════════════════

def AugmentSample(arr: np.ndarray,
                  NoiseFrac:  float = 0.05,
                  ScaleRange: float = 0.10,
                  TimeRange:  float = 0.30,
                  rng: np.random.Generator = None) -> np.ndarray:
   
    if rng is None:
        rng = np.random.default_rng()   # random seed each time

    arr  = arr.astype(np.float32, copy=True)
    T, F = arr.shape

    # Additive noise — proportional to each feature's own std
    PerSensorStd = arr.std(axis=0, keepdims=True)
    noise        = rng.standard_normal(size=arr.shape) * PerSensorStd * NoiseFrac
    arr         += noise.astype(np.float32)

    # Per-feature scaling — simulates slight sensor variation
    scale = rng.uniform(1.0 - ScaleRange, 1.0 + ScaleRange, size=(1, F)).astype(np.float32)
    arr  *= scale

    # Time warp — stretch or compress the sequence slightly
    timewarp = rng.uniform(1.0 - TimeRange, 1.0 + TimeRange)
    NewT     = max(4, int(round(T * timewarp)))

    if NewT != T:
        old_x = np.arange(T,dtype=np.float32)
        new_x = np.linspace(0, T - 1, NewT, dtype=np.float32)
        f     = interp1d(old_x, arr, axis=0, kind="linear")
        arr   = f(new_x).astype(np.float32)               # (NewT, 16)
        arr = resample(arr, T, axis=0).astype(np.float32) # (T, 16)

    return arr   # (T, 16) — same shape as input, always

#=====================================
# STEP 7 — Z-SCORE Normalization 
#=====================================

def compute_signer_stats(all_recordings):
    """
    all_recordings_of_signer: list of arrays, each shape (T, 16)
                              ALL recordings from ONE signer
    returns: mean (16,), std (16,)

    Call this ONLY on training data — never include validation recordings.
    Save the returned mean and std — needed for inference on Raspberry Pi
    """
    stacked = np.vstack(all_recordings)         # (N*T, 16)
    mean    = stacked.mean(axis=0)                        # (16,)
    std     = stacked.std(axis=0)                         # (16,)
    std[std == 0] = 1.0 # Prevent division by zero 
    return mean, std

def apply_zscore(recording, mean, std):
    """
    recording: (T, 16) — processed recording
    mean:      (16,)   
    std:       (16,)   
    returns:   (T, 16) — normalized recording

    Training  → mean/std computed from all training recordings
    Inference → mean/std loaded from saved .npy files
    """
    return (recording - mean) / std # (T, 16)


#========================================
# COMBINED PREPROCESSING 
#========================================
def preprocess_single(recording, target_length=100):
    """
    Applies steps 1-5 to one raw recording.
    Input:  (T, 11)          — raw sensor data from .npy file
    Output: (target_length, 16) — clean, fixed-length features
    """
    rec = remove_noise(recording)     #(T,11)                                # (T, 11)
    resting = get_resting_pose(rec, n_resting_rows=120) 
    rec = subtract_resting_pose(rec, resting)    #(T,11)                 # (T, 11)
    rec = compute_relational_features(rec)      #(T,8)                      # (T, 8)
    rec = normalize_length(rec, target_length) #(100,8)                       # (100, 8)
    rec = compute_delta(rec)          #(100,16)                                # (100, 16)
    return rec   
