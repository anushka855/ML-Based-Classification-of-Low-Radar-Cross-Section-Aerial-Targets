from src.simulation import simulate_target
from src.tracking import run_tracker
from src.feature_extraction import extract_features


def run_class(target_type):

    truth_states, detections, transition_model, measurement_model = simulate_target(
        target_type=target_type
    )

    track = run_tracker(
        truth_states,
        detections,
        transition_model,
        measurement_model
    )

    features = extract_features(track, detections)

    print(f"\n===== {target_type.upper()} =====")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":

    run_class("bird")
    run_class("aircraft")
    run_class("stealth")
