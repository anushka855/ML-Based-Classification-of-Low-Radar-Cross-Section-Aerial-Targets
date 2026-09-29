''' .csv for smoke testing [since human readable] and .npz for actual ML training '''

import numpy as np
from feature_extraction import extract_features_timestep
from feature_extraction import export_csv
from simulation import simulate_scene
from tracking import CVTracker, CTTracker, IMMTracker

# -------------------------------------------------
# Label mapping
# -------------------------------------------------

LABEL_MAP = {
    "empty": 0,
    "aircraft": 1,
    "stealth": 2
}


# -------------------------------------------------
# Generate dataset from one scene
# -------------------------------------------------

'''Manual tracking loop required to extract timestep-wise features; 
run_tracking() only provides final aggregated results and is built for plotting rather than feature extraction.

so run_tracking() ditched for dataset generation and rather re-implemnetation with logging done. 
'''
def generate_scene_dataset(scene_type, num_steps=80):

    scene = simulate_scene(scene_type, num_steps=num_steps, plot=False)

    cv = CVTracker(
        dt=scene["metadata"]["dt"],
        measurement_model=scene["measurements"]["measurement_model"],
        process_noise_scale=2
    )

    ct = CTTracker(
        dt=scene["metadata"]["dt"],
        measurement_model=scene["measurements"]["measurement_model"]
    )

    imm = IMMTracker(cv, ct)

    X_scene = []
    y_scene = []

    label = LABEL_MAP[scene_type]

    detections_all = scene["measurements"]["detections"]

    for t, detections in enumerate(detections_all):

        # --- initialization phase ---
        if not cv.initialized:
            cv.step(detections)

        if not ct.initialized:
            ct.step(detections)

        # --- IMM tracking phase ---
        if cv.initialized and ct.initialized:

            imm.interaction()

            cv.predict()
            ct.predict()

            gated_cv = cv.gate(detections)
            logL_cv = cv.update(gated_cv)

            gated_ct = ct.gate(detections)
            logL_ct = ct.update(gated_ct)

            imm.update_mode_probabilities(logL_cv, logL_ct)
            imm.fuse()

            # logging
            cv.estimate_history.append(cv.state.state_vector.flatten().copy())
            cv.cov_trace_history.append(np.trace(cv.state.covar))
            cv.status_history.append(cv.status)

            ct.estimate_history.append(ct.state.state_vector.flatten().copy())
            ct.cov_trace_history.append(np.trace(ct.state.covar))
            ct.status_history.append(ct.status)

        result = {
            "scene": scene,
            "cv_tracker": cv,
            "ct_tracker": ct,
            "imm_tracker": imm
        }

        features = extract_features_timestep(result, t)

        X_scene.append(features)
        y_scene.append(label)

    return np.array(X_scene), np.array(y_scene)
# -------------------------------------------------
# Build full dataset
# -------------------------------------------------

def build_dataset(
        aircraft_scenes=50,
        stealth_scenes=50,
        empty_scenes=50,
        num_steps=80):

    X_all = []
    y_all = []
    scene_ids = []
    scene_counter = 0

    # Aircraft scenes
    print("Aircraft Scenes:")
    for i in range(aircraft_scenes):
        X, y = generate_scene_dataset("aircraft", num_steps)
        X_all.append(X)
        y_all.append(y)

        scene_ids.extend([scene_counter] * len(y))
        scene_counter += 1
        print(f"aircraft scene {i} scene_id={scene_counter-1}")

    # Stealth scenes
    print("Stealth Scenes:")
    for i in range(stealth_scenes):
        X, y = generate_scene_dataset("stealth", num_steps)
        X_all.append(X)
        y_all.append(y)

        scene_ids.extend([scene_counter] * len(y))
        scene_counter += 1
        print(f"stealth scene {i} scene_id={scene_counter-1}")

    # Empty scenes
    print("Empty Scenes:")
    for i in range(empty_scenes):
        X, y = generate_scene_dataset("empty", num_steps)
        X_all.append(X)
        y_all.append(y)

        scene_ids.extend([scene_counter] * len(y))
        scene_counter += 1
        print(f"empty scene {i} scene_id={scene_counter-1}")

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)
    scene_ids = np.array(scene_ids)

    return X_all, y_all, scene_ids

# -------------------------------------------------
# Save dataset
# -------------------------------------------------
def save_dataset(X, y, scene_ids, filename="radar_dataset.npz"):

    np.savez_compressed(
        filename,
        X=X,
        y=y,
        scene_ids=scene_ids
    )

    print("Dataset saved:", filename)
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])

# -------------------------------------------------
# Load dataset
# -------------------------------------------------

def load_dataset(filename="radar_dataset.npz"):

    data = np.load(filename)

    X = data["X"]
    y = data["y"]
    scene_ids = data["scene_ids"]

    return X, y, scene_ids

# -------------------------------------------------
# Main (dataset generation)
# -------------------------------------------------
if __name__ == "__main__":

    X, y, scene_ids = build_dataset(
        aircraft_scenes=50,
        stealth_scenes=50,
        empty_scenes=50
    )

    save_dataset(X, y, scene_ids)
    #generate csv only for smoke testing. 
    #export_csv(X, y)