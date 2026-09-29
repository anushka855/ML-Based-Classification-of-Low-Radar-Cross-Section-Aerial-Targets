import joblib
import numpy as np
import matplotlib.pyplot as plt

from tracking import run_tracking
from feature_extraction import extract_features_timestep


# -------------------------------------------------
# Load trained classifier
# -------------------------------------------------
model = joblib.load("results/radar_classifier.pkl")
label_map = {
    0: "Empty",
    1: "Aircraft",
    2: "Stealth"
}

scene_types = ["empty", "aircraft", "stealth"]


# -------------------------------------------------
# Run all scenes
# -------------------------------------------------

for scene_type in scene_types:

    print("\n==============================")
    print("Running scene:", scene_type)
    print("==============================\n")

    result = run_tracking(scene_type)

    scene = result["scene"]
    num_steps = scene["metadata"]["num_steps"]

    prob_empty = []
    prob_aircraft = []
    prob_stealth = []

    predictions = []

    # -------------------------------------------------
    # Scan-by-scan classification
    # -------------------------------------------------

    for t in range(num_steps):

        features = extract_features_timestep(result, t)

        X = features.reshape(1, -1)

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        predictions.append(prediction)

        prob_empty.append(probabilities[0])
        prob_aircraft.append(probabilities[1])
        prob_stealth.append(probabilities[2])

        print(
            f"Scan {t:02d} | "
            f"Prediction: {label_map[prediction]} | "
            f"P(empty)={probabilities[0]:.2f} "
            f"P(aircraft)={probabilities[1]:.2f} "
            f"P(stealth)={probabilities[2]:.2f}"
        )

    prob_empty = np.array(prob_empty)
    prob_aircraft = np.array(prob_aircraft)
    prob_stealth = np.array(prob_stealth)

    # -------------------------------------------------
    # Plot confidence evolution
    # -------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.plot(prob_empty, label="P(Empty)", linewidth=2)
    plt.plot(prob_aircraft, label="P(Aircraft)", linewidth=2)
    plt.plot(prob_stealth, label="P(Stealth)", linewidth=2)

    plt.xlabel("Radar Scan Index")
    plt.ylabel("Classification Probability")

    plt.title(f"Online Target Classification Confidence — {scene_type}")

    plt.legend()
    plt.grid(True)

    plt.show()