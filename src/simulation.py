import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection


# ============================================================
# Target Configuration
# ============================================================


TARGET_CONFIG = {
    "bird": {"speed": 50, "process_noise": 2.0, "rcs": 0.01},
    "aircraft": {"speed": 250.0, "process_noise": 0.1, "rcs": 5.0},
    "stealth": {"speed": 250.0, "process_noise": 0.1, "rcs": 0.01},
    "empty": None
}

MANEUVER_CONFIG = {
    "aircraft": {
        "maneuver_prob": 0.05,
        "maneuver_min_deg": 3,
        "maneuver_max_deg": 6,
        "maneuver_duration_min": 8,
        "maneuver_duration_max": 12
    },
    "stealth": {
        "maneuver_prob": 0.04,
        "maneuver_min_deg": 4,
        "maneuver_max_deg": 8,
        "maneuver_duration_min": 6,
        "maneuver_duration_max": 10
    },
    "bird": {
        "maneuver_prob": 0.4,
        "maneuver_min_deg": 5,
        "maneuver_max_deg": 25,
        "maneuver_duration_min": 1,
        "maneuver_duration_max": 4
    }
}

# ============================================================
# Radar Configuration
# ============================================================

RADAR_CONFIG = {
    "Pt": 1e9,
    "noise_floor": 1e-9,
    "bearing_std_deg": 1.6,
    "range_std_m": 20.0,
    "max_range": 15000,

    "num_range_bins": 200,
    "num_bearing_bins": 180,

    # CFAR
    "cfar_training": 8,
    "cfar_guard": 3,
    "cfar_pfa": 1e-4
}


# ============================================================
# Main Simulation
# ============================================================

def simulate_scene(scene_type="aircraft",
                   num_steps=80,
                   dt=1.0,
                   plot=False):

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([
            np.radians(RADAR_CONFIG["bearing_std_deg"])**2,
            RADAR_CONFIG["range_std_m"]**2
        ])
    )

    config = TARGET_CONFIG.get(scene_type)
    has_target = config is not None

    truth = None

    # ------------------------------------------------------------
    # Inbound target from fixed 10 km
    # ------------------------------------------------------------

    if has_target:

        initial_range = 15000.0
        initial_bearing = np.random.uniform(-np.pi/6, np.pi/6)

        x0 = initial_range * np.cos(initial_bearing)
        y0 = initial_range * np.sin(initial_bearing)

        dx = -x0
        dy = -y0
        norm = np.sqrt(dx**2 + dy**2)

        vx = config["speed"] * dx / norm
        vy = config["speed"] * dy / norm

        truth = GaussianState(
            StateVector([x0, vx, y0, vy]),
            np.diag([1,1,1,1]),
            timestamp=datetime.now()
        )
    truth_states = []
    detections = []

    truth_x, truth_y = [], []
    clutter_x_all, clutter_y_all = [], []
    detect_x_all, detect_y_all = [], []

    # NEW: detection logging for ML
    detection_positions = []      # positions per scan
    detection_snr_values = []     # SNR per scan

    snr_history = []
    clutter_count_history = []


    time_history = []
    range_history = []
    radial_velocity_history = []
    detection_count_history = []
    detection_presence_history = []


    maneuver_steps_remaining = 0

    max_range = RADAR_CONFIG["max_range"]
    Nr = RADAR_CONFIG["num_range_bins"]
    Nb = RADAR_CONFIG["num_bearing_bins"]

    range_bins = np.linspace(0, max_range, Nr)
    bearing_bins = np.linspace(-np.pi, np.pi, Nb)

    for step in range(num_steps):

        if has_target:
            truth_states.append(truth)

            x = truth.state_vector[0, 0]
            y = truth.state_vector[2, 0]
            vx = truth.state_vector[1, 0]
            vy = truth.state_vector[3, 0]

            R = np.sqrt(x**2 + y**2) + 1e-6
            B = np.arctan2(y, x)
        else:
            x = y = vx = vy = 0.0
            R = np.random.uniform(1000, max_range)
            B = np.random.uniform(-np.pi, np.pi)

        time_history.append(step * dt)
        range_history.append(R)


        truth_x.append(x)
        truth_y.append(y)

        power_map = np.ones((Nr, Nb)) * RADAR_CONFIG["noise_floor"]

        # ============================================================
        # TARGET
        # ============================================================
        if has_target:

            rcs_fluct = np.random.exponential(scale=config["rcs"])
            target_power = (RADAR_CONFIG["Pt"] * rcs_fluct) / (R**4)

            snr = target_power / RADAR_CONFIG["noise_floor"]
            snr_history.append(10 * np.log10(snr + 1e-12))

            r_idx = np.argmin(np.abs(range_bins - R))
            b_idx = np.argmin(np.abs(bearing_bins - B))

            power_map[r_idx, b_idx] += target_power
        else:
            snr_history.append(-40)
        # ============================================================
        # CLUTTER
        # ============================================================

        clutter_rate = 2 + 10 * (R / max_range)**2
        clutter_count = np.random.poisson(clutter_rate)
        clutter_count_history.append(clutter_count)

        # ============================================================
        # CLUTTER - realistic patches + speckle
        # ============================================================

        # base speckle (exponential / Rayleigh-like) — stochastic noise floor
        noise_map = np.random.exponential(scale=RADAR_CONFIG["noise_floor"], size=(Nr, Nb))
        power_map += noise_map

        # add a few patchy clutter regions (2D Gaussian blobs in range-bearing bins)
        num_patches = np.random.randint(4, 10)  # how many clutter patches this scan
        for _p in range(num_patches):
            # center in physical coordinates
            cr = max_range * np.sqrt(np.random.rand())         # range (area-consistent)
            cb = np.random.vonmises(0, 0.0) if False else np.random.uniform(-np.pi, np.pi)
            # amplitude: heavier tail but capped
            amp = np.random.gamma(shape=1.5, scale=0.6)  # modifies local RCS of patch

            # convert to bin centers
            r_c = np.argmin(np.abs(range_bins - cr))
            b_c = np.argmin(np.abs(bearing_bins - cb))

            # patch widths in bins (scale with range for realism)
            sigma_r = max(2, int(5 + (r_c / Nr) * 10))      # wider patches at larger ranges
            sigma_b = max(1, int(2 + (b_c / Nb) * 4))

            # create local 2D Gaussian bump and add it
            r_indices = np.arange(max(0, r_c - 4*sigma_r), min(Nr, r_c + 4*sigma_r))
            b_indices = np.arange(max(0, b_c - 4*sigma_b), min(Nb, b_c + 4*sigma_b))

            rr, bb = np.meshgrid(r_indices, b_indices, indexing='ij')
            dr = (rr - r_c) / max(1.0, sigma_r)
            db = (bb - b_c) / max(1.0, sigma_b)
            bump = amp * np.exp(-0.5 * (dr**2 + db**2))

            # scale bump by inverse-range^4 effect at patch center (approx)
            patch_range = range_bins[r_c] + 1e-6
            bump *= (RADAR_CONFIG["Pt"] / (patch_range**4))

            power_map[r_indices[:, None], b_indices[None, :]] += bump

            # store some example clutter scatter points for plotting
            # sample a few points from the bump to keep scatter realistic
            flat_idx = np.argwhere(bump > 0)
            if flat_idx.size:
                samples = flat_idx[np.random.choice(flat_idx.shape[0],
                                                    size=min(4, flat_idx.shape[0]),
                                                    replace=False)]
                for s in samples:
                    rr_s = int(rr[s[0], s[1]])
                    bb_s = int(bb[s[0], s[1]])
                    cr_phys = range_bins[rr_s]
                    cb_phys = bearing_bins[bb_s]
                    clutter_x_all.append(cr_phys * np.cos(cb_phys))
                    clutter_y_all.append(cr_phys * np.sin(cb_phys))

        # Add a small number of independent point-clutter hits (like spiky birds)
        point_clutter_count = np.random.poisson(3)
        for _ in range(point_clutter_count):
            cr = max_range * np.sqrt(np.random.rand())
            cb = np.random.uniform(-np.pi, np.pi)
            clutter_rcs = np.random.gamma(shape=1.0, scale=0.4)
            clutter_power = (RADAR_CONFIG["Pt"] * clutter_rcs) / (cr**4 + 1e-6)
            r_i = np.argmin(np.abs(range_bins - cr))
            b_i = np.argmin(np.abs(bearing_bins - cb))
            power_map[r_i, b_i] += clutter_power
            clutter_x_all.append(cr * np.cos(cb))
            clutter_y_all.append(cr * np.sin(cb))

        # ============================================================
        # 2D CA-CFAR
        # ============================================================

        T = RADAR_CONFIG["cfar_training"]
        G = RADAR_CONFIG["cfar_guard"]
        Pfa = RADAR_CONFIG["cfar_pfa"]

        total_window = ((2*(T+G)+1)**2)
        guard_window = (2*G+1)**2
        N = total_window - guard_window

        alpha = N * (Pfa**(-1/N) - 1)

        detected_cells = []

        for i in range(T+G, Nr-T-G):
            for j in range(T+G, Nb-T-G):

                cut_power = power_map[i, j]

                window = power_map[i-T-G:i+T+G+1,
                                   j-T-G:j+T+G+1]

                guard = power_map[i-G:i+G+1,
                                  j-G:j+G+1]

                training_cells = window.flatten()

                #*** OS-CFAR Implementation !! *** 
                # remove guard cells
                guard_mask = np.zeros_like(window, dtype=bool)
                guard_mask[G:-G or None, G:-G or None] = True
                training_cells = training_cells[~guard_mask.flatten()]

                # Flatten window and remove guard cells
                window_flat = window.flatten()
                # Build guard mask (same shape as window)
                gmask = np.zeros_like(window, dtype=bool)
                # mark guard cells True inside center (guard block)
                gmask[G:-G or None, G:-G or None] = True
                training_flat = window_flat[~gmask.flatten()]

                # If too few training cells skip
                if training_flat.size < max(3, int(0.5 * N)):
                    continue

                # Ordered-statistic CFAR: pick k-th order statistic as noise estimate
                k = int(0.75 * training_flat.size)  # 70th percentile cell   [ keep 70- 75]
                sorted_cells = np.sort(training_flat)
                noise_est_os = sorted_cells[min(k, len(sorted_cells)-1)]

                # OS-CFAR will require manual tuning !!!!!!!
                alpha_os = 5 # start tuning manually
                threshold = alpha_os * noise_est_os

                if cut_power > threshold:
                    detected_cells.append((i, j))

        # ============================================================
        # LOCAL PEAK FILTER (stable clustering)
        # ============================================================

        filtered = []

        for (i, j) in detected_cells:

            local = power_map[i-1:i+2, j-1:j+2]
            if power_map[i, j] == np.max(local):
                filtered.append((i, j))

        detected_cells = filtered

        # ============================================================
        # MEASUREMENT GENERATION
        # ============================================================

        step_detections = []

        # NEW buffers for this scan
        scan_positions = []
        scan_snrs = []
        target_vr = (x*vx + y*vy) / (R + 1e-6)

        radial_velocity_history.append(target_vr)


        for i, j in detected_cells:

            det_range = range_bins[i]
            det_bearing = bearing_bins[j]

            snr_linear = power_map[i, j] / RADAR_CONFIG["noise_floor"]
            snr_db = 10*np.log10(snr_linear + 1e-12)

            if snr_db < 10:
                continue

            dx = det_range * np.cos(det_bearing)
            dy = det_range * np.sin(det_bearing)

            vr_est = (dx*vx + dy*vy) / (det_range + 1e-6)

            # ------------------------------------------------------------
            # Adaptive Doppler gate (defense-style behavior)
            # ------------------------------------------------------------

            base_gate = 50.0          # normal Doppler gate (m/s)
            min_gate = 120.0          # widened gate near zero crossing
            zero_cross_band = 60.0    # region where widening occurs

            if abs(target_vr) < zero_cross_band:
                # Linearly widen gate as vr approaches 0
                scale = 1.0 - abs(target_vr) / zero_cross_band
                doppler_gate = base_gate + scale * (min_gate - base_gate)
            else:
                doppler_gate = base_gate

            if abs(vr_est - target_vr) > doppler_gate:
                continue

            range_std = RADAR_CONFIG["range_std_m"] * \
                        np.sqrt(10 / max(snr_linear, 1e-6))
            bearing_std = np.radians(RADAR_CONFIG["bearing_std_deg"]) * \
                          np.sqrt(10 / max(snr_linear, 1e-6))

            noisy_range = det_range + np.random.normal(0, range_std)
            noisy_bearing = det_bearing + np.random.normal(0, bearing_std)

            x_det = noisy_range * np.cos(noisy_bearing)
            y_det = noisy_range * np.sin(noisy_bearing)

            detect_x_all.append(x_det)
            detect_y_all.append(y_det)

            # NEW: store for ML feature extraction
            scan_positions.append((x_det, y_det))
            scan_snrs.append(snr_db)

            step_detections.append(
                Detection(
                    StateVector([noisy_bearing, noisy_range]),
                    timestamp=datetime.now() + timedelta(seconds=step*dt),
                    measurement_model=measurement_model,
                    metadata={"snr_linear": snr_linear}
                )
            )

        detections.append(step_detections)

        # NEW logging
        detection_positions.append(scan_positions)
        detection_snr_values.append(scan_snrs)
        detection_count = len(step_detections)
        detection_count_history.append(detection_count)
        detection_presence_history.append(detection_count > 0)


        # ============================================================
        # MOTION PROPAGATION (unchanged)
        # ============================================================
        if has_target:

            speed = np.sqrt(vx**2 + vy**2)
            heading = np.arctan2(vy, vx)

            maneuver_cfg = MANEUVER_CONFIG[scene_type]

            if maneuver_steps_remaining == 0:
                if np.random.rand() < maneuver_cfg["maneuver_prob"]:
                    maneuver_steps_remaining = np.random.randint(
                        maneuver_cfg["maneuver_duration_min"],
                        maneuver_cfg["maneuver_duration_max"]
                    )

            if maneuver_steps_remaining > 0:
                heading += np.radians(np.random.uniform(-20, 20))
                maneuver_steps_remaining -= 1

            if scene_type == "bird":
                speed += np.random.normal(0, 2.0)
                speed = max(speed, 5.0)

            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)

            vx += np.random.normal(0, config["process_noise"])
            vy += np.random.normal(0, config["process_noise"])

            x += vx * dt
            y += vy * dt

            truth = GaussianState(
                StateVector([x, vx, y, vy]),
                truth.covar,
                timestamp=truth.timestamp + timedelta(seconds=dt)
            )
    

    
    scene_data = {
        
        "metadata": {
            "scene_type": scene_type,
            "num_steps": num_steps,
            "dt": dt,
            "target_config": TARGET_CONFIG.get(scene_type),
            "radar_config": RADAR_CONFIG,
            "maneuver_config": MANEUVER_CONFIG.get(scene_type)
        },

        "truth": {
            "states": truth_states,
            "x": np.array(truth_x),
            "y": np.array(truth_y),
            "range": np.array(range_history),
            "radial_velocity": np.array(radial_velocity_history),
            "time": np.array(time_history)
        },

        "radar_metrics": {
            "snr_db": np.array(snr_history),
            "clutter_count": np.array(clutter_count_history),
            "detection_count": np.array(detection_count_history),
            "detection_presence": np.array(detection_presence_history)
        },

        "measurements": {
            "detections": detections,
            "measurement_model": measurement_model,

            # NEW for ML
            "detection_positions": detection_positions,
            "detection_snrs": detection_snr_values
        },

        "plot_data": {
            "clutter_x": clutter_x_all,
            "clutter_y": clutter_y_all,
            "detect_x": detect_x_all,
            "detect_y": detect_y_all
        }
    }
    return scene_data




def plot_scene(scene_data):
    truth_x = scene_data["truth"]["x"]
    truth_y = scene_data["truth"]["y"]
    clutter_x_all = scene_data["plot_data"]["clutter_x"]
    clutter_y_all = scene_data["plot_data"]["clutter_y"]
    detect_x_all = scene_data["plot_data"]["detect_x"]
    detect_y_all = scene_data["plot_data"]["detect_y"]
    snr_history = scene_data["radar_metrics"]["snr_db"]
    clutter_count_history = scene_data["radar_metrics"]["clutter_count"]
    scene_type = scene_data["metadata"]["scene_type"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Radar Diagnostics - {scene_type}", fontsize=14)

    if scene_type == "bird":
        margin = 5000
        axs[0, 0].set_xlim(min(truth_x)-margin, max(truth_x)+margin)
        axs[0, 0].set_ylim(min(truth_y)-margin, max(truth_y)+margin)

    axs[0, 0].plot(truth_x, truth_y, linewidth=2, label="Truth")
    axs[0, 0].scatter(clutter_x_all, clutter_y_all,
                      s=6, alpha=0.15, label="Clutter")
    axs[0, 0].scatter(detect_x_all, detect_y_all,
                      s=20, marker='x', label="All Detections")
    axs[0, 0].legend()
    axs[0, 0].set_title("Full Scene (Anonymous Detections)")

    axs[0, 1].plot(snr_history)
    axs[0, 1].set_title("Target SNR (dB)")

    axs[1, 0].plot(clutter_count_history)
    axs[1, 0].set_title("Clutter Count Per Scan")

    heatmap, _, _ = np.histogram2d(clutter_x_all,
                                   clutter_y_all,
                                   bins=50)
    axs[1, 1].imshow(heatmap.T, origin='lower')
    axs[1, 1].set_title("Clutter Density Map")

    for ax in axs.flat:
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for scene_type in ["aircraft", "stealth", "empty"]:
        scene_data = simulate_scene(scene_type, plot=False)

        #plotting
        plot_scene(scene_data)
