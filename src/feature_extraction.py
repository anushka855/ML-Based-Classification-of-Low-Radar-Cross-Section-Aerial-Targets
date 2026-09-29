''' .csv for smoke testing [since human readable] and .npz for actual ML training '''

import numpy as np
import pandas as pd
from tracking import run_tracking

FEATURE_NAMES = [
    "mean_snr",
    "snr_variance",
    "snr_trend",

    "total_detections",
    "detections_per_scan_std",
    "max_detections_in_scan",
    "longest_detection_gap",
    "detection_presence_ratio",
    "detection_burst_count",

    "centroid_motion_speed",
    "centroid_motion_variance",

    "track_initialized",
    "time_to_track_initialization",
    "track_length",
    "covariance_trace_mean",
    "covariance_trace_growth",
    "covariance_trace_std",

    "mean_CV_probability",
    "mean_CT_probability",
    "mode_switch_count",
    "mode_probability_variance",

    "gated_detections_mean",
    "tracker_miss_ratio",
    "innovation_magnitude_mean",
    "detections_inside_gate_ratio",
    "snr_dropout_ratio",
    "track_velocity_variance"
]

# ---------------- Utility helpers ----------------

def safe_mean(x):
    if len(x) == 0:
        return 0.0
    return float(np.mean(x))


def safe_var(x):
    if len(x) < 2:
        return 0.0
    return float(np.var(x))


def snr_trend(snr_series):
    if len(snr_series) < 3:
        return 0.0

    x = np.arange(len(snr_series))
    y = np.array(snr_series)

    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def longest_zero_run(binary_series):

    longest = 0
    current = 0

    for v in binary_series:
        if v == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def burst_count(binary_series):

    bursts = 0
    prev = 0

    for v in binary_series:
        if v == 1 and prev == 0:
            bursts += 1
        prev = v

    return bursts


def centroid_motion(detection_positions):

    centroids = []

    for scan in detection_positions:
        if len(scan) == 0:
            continue

        xs = [p[0] for p in scan]
        ys = [p[1] for p in scan]

        centroids.append((np.mean(xs), np.mean(ys)))

    if len(centroids) < 2:
        return 0.0, 0.0

    speeds = []

    for i in range(1, len(centroids)):
        dx = centroids[i][0] - centroids[i-1][0]
        dy = centroids[i][1] - centroids[i-1][1]
        speeds.append(np.sqrt(dx**2 + dy**2))

    return safe_mean(speeds), safe_var(speeds)


# ---------------- Core feature extractor ----------------

def extract_features_timestep(result, t):

    scene = result["scene"]
    cv = result["cv_tracker"]
    ct = result["ct_tracker"]
    imm = result["imm_tracker"]

    radar = scene["radar_metrics"]
    measurements = scene["measurements"]

    snr_series = radar["snr_db"][:t+1]
    detection_count = radar["detection_count"][:t+1]
    detection_presence = radar["detection_presence"][:t+1]

    detection_positions = measurements["detection_positions"][:t+1]

    # ------------------------------------------------
    # Radar signal
    # ------------------------------------------------

    mean_snr = safe_mean(snr_series)
    snr_variance = safe_var(snr_series)
    snr_tr = snr_trend(snr_series)
    # SNR dropout ratio (fraction of scans with very low SNR)
    dropout_threshold = -20
    snr_dropout_ratio = float(np.mean(np.array(snr_series) < dropout_threshold))

    # ------------------------------------------------
    # Detection statistics
    # ------------------------------------------------

    total_detections = int(np.sum(detection_count))
    detections_per_scan_std = float(np.std(detection_count))
    max_detections_in_scan = int(np.max(detection_count)) if len(detection_count) else 0

    longest_gap = longest_zero_run(detection_presence)
    detection_presence_ratio = safe_mean(detection_presence)
    detection_bursts = burst_count(detection_presence)

    # ------------------------------------------------
    # Spatial coherence
    # ------------------------------------------------

    centroid_speed, centroid_var = centroid_motion(detection_positions)

    # ------------------------------------------------
    # Tracker behaviour
    # ------------------------------------------------

    init_time = cv.initialization_time

    if init_time is None:
        track_initialized = 0
        time_to_init = -1
    else:
        init_scan = 2   # tracker initializes after 2 detections

        if t < init_scan:
            track_initialized = 0
            time_to_init = -1
        else:
            track_initialized = 1
            time_to_init = t - init_scan

    track_length = min(t + 1, len(cv.cov_trace_history))
    cov_slice = cv.cov_trace_history[:track_length]

    if track_initialized == 0 or track_length == 0:
        cov_mean = 0
        cov_growth = 0
        cov_std = 0
    else:
        cov_mean = np.log1p(safe_mean(cov_slice))

        if track_length > 1:
            cov_growth = np.log1p(abs(cov_slice[-1] - cov_slice[0]))
            cov_std = np.log1p(float(np.std(cov_slice)))
        else:
            cov_growth = 0
            cov_std = 0


    # ------------------------------------------------
    # Track velocity stability
    # ------------------------------------------------

    velocity_variance = 0

    if len(cv.estimate_history) > 1:

        track_states = np.array(cv.estimate_history[:t+1])

        vx = track_states[:,1]
        vy = track_states[:,3]

        speeds = np.sqrt(vx**2 + vy**2)

        if len(speeds) > 1:
            velocity_variance = float(np.var(speeds))
    # ------------------------------------------------
    # IMM behaviour
    # ------------------------------------------------

    mu = np.array(imm.mu_history[:t+1])

    if len(mu) > 0:
        mean_cv_prob = safe_mean(mu[:,0])
        mean_ct_prob = safe_mean(mu[:,1])
        mode_std = float(np.std(mu[:,0]))
    else:
        mean_cv_prob = 0
        mean_ct_prob = 0
        mode_std = 0

    mode_history = np.array(imm.mode_history[:t+1])

    mode_switch_count = np.sum(mode_history[1:] != mode_history[:-1])
    # ------------------------------------------------
    # Detection–tracker interaction
    # ------------------------------------------------

    gated_counts = cv.gated_count_history[:t+1]
    gated_mean = safe_mean(gated_counts)

    misses = cv.miss_history[:t+1]
    miss_ratio = safe_mean(misses)

    innovations = cv.innovation_history[:t+1]
    innovation_mean = safe_mean(innovations)

    mean_detections = safe_mean(detection_count)

    if mean_detections == 0:
        detections_inside_gate_ratio = 0
    else:
        detections_inside_gate_ratio = min(1.0, gated_mean / mean_detections)
    # ------------------------------------------------
    # Feature vector
    # ------------------------------------------------

    features = np.array([
        mean_snr,
        snr_variance,
        snr_tr,

        total_detections,
        detections_per_scan_std,
        max_detections_in_scan,
        longest_gap,
        detection_presence_ratio,
        detection_bursts,

        centroid_speed,
        centroid_var,

        track_initialized,
        time_to_init,
        track_length,
        cov_mean,
        cov_growth,
        cov_std,

        mean_cv_prob,
        mean_ct_prob,
        mode_switch_count,
        mode_std,

        gated_mean,
        miss_ratio,
        innovation_mean,
        detections_inside_gate_ratio,
        snr_dropout_ratio,
        velocity_variance
        ])

    return features



# ----------------- .csv export helper  -----------------
def export_csv(X, y, filename="radar_dataset_debug.csv"):

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y

    df.to_csv(filename, index=False)

    print("CSV exported:", filename)