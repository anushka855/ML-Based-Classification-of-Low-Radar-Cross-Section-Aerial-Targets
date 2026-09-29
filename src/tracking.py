import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity
)
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.models.transition.nonlinear import ConstantTurn
from simulation import simulate_scene
from scipy.special import logsumexp
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.transition.nonlinear import ConstantTurn

class CVTracker:

    def __init__(self, dt, measurement_model, process_noise_scale=4.0):

        self.dt = dt
        self.measurement_model = measurement_model

        self.transition_model = CombinedLinearGaussianTransitionModel([
            ConstantVelocity(process_noise_scale),
            ConstantVelocity(process_noise_scale)
        ])

        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(measurement_model)

        self.base_bearing_std = np.radians(1.6)
        self.base_range_std = 20.0
        self.snr_ref = 10.0

        self.prob_detection = 0.85
        self.clutter_density = 1e-5

        self.state = None
        self.initialized = False
        self.status = "tentative"

        self.consecutive_misses = 0
        self.hit_count = 0
        self.init_buffer = []

        # Logging
        self.estimate_history = []
        self.cov_trace_history = []
        self.status_history = []

        # --- ML logging ---
        self.gated_count_history = []
        self.innovation_history = []
        self.miss_history = []
        self.initialization_time = None

    # ---------------- Prediction ----------------

    def predict(self):
        new_timestamp = self.state.timestamp + timedelta(seconds=self.dt)
        self.state = self.predictor.predict(self.state, timestamp=new_timestamp)

    # ---------------- Build dynamic measurement model ----------------

    def build_dynamic_model(self, det):
        snr_linear = det.metadata.get("snr_linear", 10.0)
        snr_linear = np.clip(snr_linear, 0.05, 1e4)

        confidence = snr_linear / (snr_linear + 2.0)

        scale = np.sqrt(self.snr_ref / snr_linear)
        scale = np.clip(scale, 0.7, 1.8)

        range_std = self.base_range_std * scale / np.sqrt(confidence)
        bearing_std = self.base_bearing_std * scale / np.sqrt(confidence)

        R_dynamic = np.diag([bearing_std**2, range_std**2])

        return type(self.measurement_model)(
            ndim_state=self.measurement_model.ndim_state,
            mapping=self.measurement_model.mapping,
            noise_covar=R_dynamic
        )

    # ---------------- Gating ----------------

    def gate(self, detections):

        if not detections:
            self.gated_count_history.append(0)
            return []

        gated = []

        for det in detections:

            dyn_model = self.build_dynamic_model(det)
            if dyn_model is None:
                continue

            meas_pred = self.updater.predict_measurement(
                self.state,
                measurement_model=dyn_model
            )

            innovation = det.state_vector - meas_pred.state_vector
            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            S = meas_pred.covar
            d2 = (innovation.T @ np.linalg.inv(S) @ innovation).item()

            if d2 < 11.83:
                gated.append(det)

        self.gated_count_history.append(len(gated))
        return gated

    # ---------------- Update (PDA) ----------------

    def update(self, gated_detections):

        PD = self.prob_detection
        lambda_c = self.clutter_density
        gamma = 11.83

        # ---------------- MISS ----------------
        if not gated_detections:

            meas_pred = self.updater.predict_measurement(self.state)
            S_pred = meas_pred.covar
            V = np.pi * gamma * np.sqrt(np.linalg.det(S_pred))

            logL = np.log((1 - PD) * lambda_c * V + 1e-12)

            self.miss_history.append(1)
            self.handle_miss()
            return logL

        # ---------------- PDA ----------------

        innovations = []
        likelihoods = []
        scan_innovations = []
        S_reference = None
        dyn_model_reference = None

        for det in gated_detections:

            dyn_model = self.build_dynamic_model(det)
            if dyn_model is None:
                continue

            meas_pred = self.updater.predict_measurement(
                self.state,
                measurement_model=dyn_model
            )

            innovation = det.state_vector - meas_pred.state_vector
            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            # --- log innovation magnitude ---
            scan_innovations.append(np.linalg.norm(innovation))

            S = meas_pred.covar
            Sinv = np.linalg.inv(S)
            d2 = (innovation.T @ Sinv @ innovation).item()

            if d2 > gamma:
                continue

            m = len(innovation)

            L = (
                1.0 /
                np.sqrt((2*np.pi)**m * np.linalg.det(S))
            ) * np.exp(-0.5 * d2)

            innovations.append(innovation)
            likelihoods.append(L)

            if S_reference is None:
                S_reference = S
                dyn_model_reference = dyn_model
        
        if len(scan_innovations) > 0:
            self.innovation_history.append(np.mean(scan_innovations))
        else:
            self.innovation_history.append(0)

        if S_reference is None:
            self.miss_history.append(1)
            self.handle_miss()
            return np.log(1e-12)
                
        likelihoods = np.array(likelihoods)

        V = np.pi * gamma * np.sqrt(np.linalg.det(S_reference))

        numerator = PD * likelihoods
        denominator = np.sum(numerator) + (1 - PD) * lambda_c * V

        betas = numerator / (denominator + 1e-12)
        beta_0 = (1 - PD) * lambda_c * V / (denominator + 1e-12)

        H = dyn_model_reference.jacobian(self.state)
        K = self.state.covar @ H.T @ np.linalg.inv(S_reference)

        innovation_bar = sum(
            betas[i] * innovations[i]
            for i in range(len(innovations))
        )

        x_new = self.state.state_vector + K @ innovation_bar

        P = self.state.covar
        I = np.eye(P.shape[0])

        P_new = beta_0 * P + (1 - beta_0) * (I - K @ H) @ P

        spread = np.zeros_like(P)
        for i in range(len(innovations)):
            diff = innovations[i] - innovation_bar
            spread += betas[i] * (K @ diff @ diff.T @ K.T)

        P_new += spread

        if np.trace(P_new) > 5e6:
            self.status = "deleted"
            self.initialized = False
            return np.log(1e-12)

        self.state.state_vector = x_new
        self.state.covar = P_new

        self.miss_history.append(0)

        self.consecutive_misses = 0
        self.hit_count += 1

        if self.status == "tentative" and self.hit_count >= 3:
            self.status = "confirmed"

        effective_likelihood = np.sum(PD * likelihoods)
        return np.log(effective_likelihood + 1e-12)

    # ---------------- Miss Handling ----------------

    def handle_miss(self):
        self.consecutive_misses += 1

        if self.consecutive_misses > 6:
            self.status = "deleted"
            self.initialized = False

    # ---------------- Step ----------------

    def step(self, detections):

        if not self.initialized:

            if detections:
                self.init_buffer.append(detections[0])

            if len(self.init_buffer) >= 2:

                d1 = self.init_buffer[-2]
                d2 = self.init_buffer[-1]

                b1, r1 = d1.state_vector.flatten()
                b2, r2 = d2.state_vector.flatten()

                x1 = r1*np.cos(b1)
                y1 = r1*np.sin(b1)

                x2 = r2*np.cos(b2)
                y2 = r2*np.sin(b2)

                vx = (x2-x1)/self.dt
                vy = (y2-y1)/self.dt

                if np.sqrt(vx**2 + vy**2) > 400:
                    self.init_buffer.pop(0)
                    return

                initial_state = StateVector([x2, vx, y2, vy])
                P = np.diag([200**2, 50**2, 200**2, 50**2])

                self.state = GaussianState(initial_state, P, timestamp=d2.timestamp)
                self.initialized = True
                self.hit_count = 2
                self.initialization_time = d2.timestamp

            return

        # ---- Normal tracking ----
        self.predict()
        gated = self.gate(detections)
        logL = self.update(gated)

        return logL

class CTTracker:

    def __init__(self, dt, measurement_model):

        self.dt = dt
        self.measurement_model = measurement_model

        self.transition_model = ConstantTurn(
            linear_noise_coeffs=[0.2, 0.2],
            turn_noise_coeff=0.02
        )
        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(measurement_model)

        self.base_bearing_std = np.radians(1.6)
        self.base_range_std = 20.0
        self.snr_ref = 10.0

        self.prob_detection = 0.85
        self.clutter_density = 1e-5

        self.state = None
        self.initialized = False
        self.status = "tentative"

        self.consecutive_misses = 0
        self.hit_count = 0
        self.init_buffer = []

        self.estimate_history = []
        self.cov_trace_history = []
        self.status_history = []

        # --- ML logging ---
        self.gated_count_history = []
        self.innovation_history = []
        self.miss_history = []
        self.initialization_time = None

    # ---------------- Prediction ----------------

    def predict(self):
        new_timestamp = self.state.timestamp + timedelta(seconds=self.dt)
        self.state = self.predictor.predict(self.state, timestamp=new_timestamp)

    # ---------------- Build dynamic model ----------------

    def build_dynamic_model(self, det):

        snr_linear = det.metadata.get("snr_linear", 10.0)
        snr_linear = np.clip(snr_linear, 0.05, 1e4)

        confidence = snr_linear / (snr_linear + 2.0)

        scale = np.sqrt(self.snr_ref / snr_linear)
        scale = np.clip(scale, 0.7, 1.8)

        range_std = self.base_range_std * scale / np.sqrt(confidence)
        bearing_std = self.base_bearing_std * scale / np.sqrt(confidence)

        R_dynamic = np.diag([bearing_std**2, range_std**2])
        
        return type(self.measurement_model)(
            ndim_state=self.measurement_model.ndim_state,
            mapping=self.measurement_model.mapping,
            noise_covar=R_dynamic
        )

    # ---------------- Gating ----------------

    def gate(self, detections):

        if not detections:
            self.gated_count_history.append(0)
            return []

        gated = []

        for det in detections:

            dyn_model = self.build_dynamic_model(det)
            if dyn_model is None:
                continue

            meas_pred = self.updater.predict_measurement(
                self.state,
                measurement_model=dyn_model
            )

            innovation = det.state_vector - meas_pred.state_vector
            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            S = meas_pred.covar
            d2 = (innovation.T @ np.linalg.inv(S) @ innovation).item()

            if d2 < 11.83:
                gated.append(det)

        self.gated_count_history.append(len(gated))
        return gated

    # ---------------- Update (PDA) ----------------

    def update(self, gated_detections):

        PD = self.prob_detection
        lambda_c = self.clutter_density
        gamma = 11.83

        if not gated_detections:

            meas_pred = self.updater.predict_measurement(self.state)
            S_pred = meas_pred.covar
            V = np.pi * gamma * np.sqrt(np.linalg.det(S_pred))

            logL = np.log((1 - PD) * lambda_c * V + 1e-12)

            self.miss_history.append(1)
            self.handle_miss()
            return logL

        innovations = []
        likelihoods = []
        scan_innovations = []
        S_reference = None
        dyn_model_reference = None

        for det in gated_detections:

            dyn_model = self.build_dynamic_model(det)
            if dyn_model is None:
                continue

            meas_pred = self.updater.predict_measurement(
                self.state,
                measurement_model=dyn_model
            )

            innovation = det.state_vector - meas_pred.state_vector
            innovation[0, 0] = np.arctan2(
                np.sin(innovation[0, 0]),
                np.cos(innovation[0, 0])
            )

            # --- log innovation magnitude ---
            scan_innovations.append(np.linalg.norm(innovation))

            S = meas_pred.covar
            Sinv = np.linalg.inv(S)
            d2 = (innovation.T @ Sinv @ innovation).item()

            if d2 > gamma:
                continue

            m = len(innovation)

            L = (
                1.0 /
                np.sqrt((2*np.pi)**m * np.linalg.det(S))
            ) * np.exp(-0.5 * d2)

            innovations.append(innovation)
            likelihoods.append(L)

            if S_reference is None:
                S_reference = S
                dyn_model_reference = dyn_model
        
        if len(scan_innovations) > 0:
            self.innovation_history.append(np.mean(scan_innovations))
        else:
            self.innovation_history.append(0)
        if S_reference is None:
            self.miss_history.append(1)
            self.handle_miss()
            return np.log(1e-12)

        likelihoods = np.array(likelihoods)

        V = np.pi * gamma * np.sqrt(np.linalg.det(S_reference))

        numerator = PD * likelihoods
        denominator = np.sum(numerator) + (1 - PD) * lambda_c * V

        betas = numerator / (denominator + 1e-12)
        beta_0 = (1 - PD) * lambda_c * V / (denominator + 1e-12)

        H = dyn_model_reference.jacobian(self.state)
        K = self.state.covar @ H.T @ np.linalg.inv(S_reference)

        innovation_bar = sum(
            betas[i] * innovations[i]
            for i in range(len(innovations))
        )

        x_new = self.state.state_vector + K @ innovation_bar

        P = self.state.covar
        I = np.eye(P.shape[0])

        P_new = beta_0 * P + (1 - beta_0) * (I - K @ H) @ P

        spread = np.zeros_like(P)
        for i in range(len(innovations)):
            diff = innovations[i] - innovation_bar
            spread += betas[i] * (K @ diff @ diff.T @ K.T)

        P_new += spread

        if np.trace(P_new) > 5e6:
            self.status = "deleted"
            self.initialized = False
            return np.log(1e-12)

        self.state.state_vector = x_new
        self.state.covar = P_new

        self.miss_history.append(0)

        self.consecutive_misses = 0
        self.hit_count += 1

        if self.status == "tentative" and self.hit_count >= 3:
            self.status = "confirmed"

        effective_likelihood = np.sum(PD * likelihoods)
        return np.log(effective_likelihood + 1e-12)

    # ---------------- Initialization ----------------

    def initialize_from_detections(self):

        d1 = self.init_buffer[-2]
        d2 = self.init_buffer[-1]

        b1, r1 = d1.state_vector.flatten()
        b2, r2 = d2.state_vector.flatten()

        x1 = r1*np.cos(b1)
        y1 = r1*np.sin(b1)

        x2 = r2*np.cos(b2)
        y2 = r2*np.sin(b2)

        vx = (x2-x1)/self.dt
        vy = (y2-y1)/self.dt

        if np.sqrt(vx**2 + vy**2) > 400:
            self.init_buffer.pop(0)
            return False

        omega = 0.0

        initial_state = StateVector([x2, vx, y2, vy, omega])

        P = np.diag([
            200**2,
            50**2,
            200**2,
            50**2,
            np.radians(20)**2
        ])

        self.state = GaussianState(initial_state, P, timestamp=d2.timestamp)
        self.initialized = True
        self.hit_count = 2
        self.initialization_time = d2.timestamp

        return True

    # ---------------- Miss Handling ----------------

    def handle_miss(self):
        self.consecutive_misses += 1

        if self.consecutive_misses > 8:
            self.status = "deleted"
            self.initialized = False

    # ---------------- Step ----------------

    def step(self, detections):

        if not self.initialized:

            if detections:
                self.init_buffer.append(detections[0])

            if len(self.init_buffer) >= 2:
                success = self.initialize_from_detections()
                if not success:
                    return

            return

        self.predict()
        gated = self.gate(detections)
        logL = self.update(gated)

        return logL    

class IMMTracker:

    def __init__(self, cv_tracker, ct_tracker):

        self.cv = cv_tracker
        self.ct = ct_tracker

        # Mode probabilities [CV, CT]
        self.mu = np.array([0.5, 0.5])

        # Mode transition matrix
        self.PI = np.array(
            [[0.95, 0.05],
            [0.05, 0.95]]
        )        
        self.mu_history = []
        self.fused_history = []

        # --- ML logging ---
        self.mode_switch_count = 0
        self.last_mode = None
        self.mode_history = []

    # ---------------- Interaction ----------------

    def interaction(self):

        if not (self.cv.initialized and self.ct.initialized):
            return

        mu_prev = self.mu
        c = self.PI.T @ mu_prev

        if np.any(c <= 1e-12):
            return

        mu_ij = np.zeros((2,2))

        for j in range(2):
            for i in range(2):
                mu_ij[i,j] = self.PI[i,j] * mu_prev[i] / c[j]

        # Expand CV state to 5D
        x_cv = self.cv.state.state_vector.flatten()
        P_cv = self.cv.state.covar

        x_cv5 = np.array([
            x_cv[0], x_cv[1],
            x_cv[2], x_cv[3],
            0.0
        ]).reshape(-1,1)

        P_cv5 = np.zeros((5,5))
        P_cv5[:4,:4] = P_cv
        P_cv5[4,4] = np.radians(10)**2

        x_ct = self.ct.state.state_vector
        P_ct = self.ct.state.covar

        X = [x_cv5, x_ct]
        P = [P_cv5, P_ct]

        mixed_states = []
        mixed_covs = []

        for j in range(2):

            # ---- Safe mixing ----
            x0 = mu_ij[0,j]*X[0] + mu_ij[1,j]*X[1]

            # If models strongly disagree on velocity direction,
            # prevent sign inversion
            v_cv = X[0][1:4:2]  # vx, vy
            v_ct = X[1][1:4:2]

            if np.dot(v_cv.flatten(), v_ct.flatten()) < 0:
                # Do NOT mix velocities, keep dominant model velocity
                dominant = np.argmax(self.mu)
                x0[1,0] = X[dominant][1,0]
                x0[3,0] = X[dominant][3,0]

            P0 = (
                mu_ij[0,j]*(P[0] + (X[0]-x0)@(X[0]-x0).T) +
                mu_ij[1,j]*(P[1] + (X[1]-x0)@(X[1]-x0).T)
            )

            mixed_states.append(x0)
            mixed_covs.append(P0)

        # Assign back
        self.cv.state.state_vector = mixed_states[0][:4]
        self.cv.state.covar = mixed_covs[0][:4,:4]

        self.ct.state.state_vector = mixed_states[1]
        self.ct.state.covar = mixed_covs[1]

    # ---------------- Mode Probability Update ----------------

    def update_mode_probabilities(self, logL_cv, logL_ct):
     
        # If one model deleted, force probability
        if not self.cv.initialized:
            self.mu = np.array([0.0, 1.0])
            self.mu_history.append(self.mu.copy())
            return

        if not self.ct.initialized:
            self.mu = np.array([1.0, 0.0])
            self.mu_history.append(self.mu.copy())
            return

        mu_prior = self.PI.T @ self.mu
        log_mu_prior = np.log(mu_prior + 1e-12)

        log_likelihoods = np.array([logL_cv, logL_ct])

        log_mu_post = log_mu_prior + log_likelihoods
        log_mu_post -= logsumexp(log_mu_post)

        self.mu = np.exp(log_mu_post)
        self.mu_history.append(self.mu.copy())
        current_mode = np.argmax(self.mu)

        if self.last_mode is not None and current_mode != self.last_mode:
            self.mode_switch_count += 1

        self.last_mode = current_mode
        self.mode_history.append(current_mode)
    # ---------------- Fusion ----------------

    def fuse(self):

        if not (self.cv.initialized and self.ct.initialized):
            return

        x_cv = self.cv.state.state_vector.flatten()
        P_cv = self.cv.state.covar

        x_ct = self.ct.state.state_vector.flatten()
        P_ct = self.ct.state.covar

        # Expand CV to 5D
        x_cv5 = np.array([
            x_cv[0], x_cv[1],
            x_cv[2], x_cv[3],
            0.0
        ])

        # State fusion
        x_fused = self.mu[0]*x_cv5 + self.mu[1]*x_ct

        # Covariance fusion (4D positional consistency)
        P_fused = (
            self.mu[0]*(P_cv + np.outer(x_cv5[:4]-x_fused[:4],
                                        x_cv5[:4]-x_fused[:4])) +
            self.mu[1]*(P_ct[:4,:4] + np.outer(x_ct[:4]-x_fused[:4],
                                               x_ct[:4]-x_fused[:4]))
        )

        self.fused_history.append(x_fused.copy())


def run_tracking(scene_type, num_steps=80):

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

    for detections in scene["measurements"]["detections"]:

        # Track initialization
        if not cv.initialized:
            cv.step(detections)

        if not ct.initialized:
            ct.step(detections)

        if not (cv.initialized and ct.initialized):
            continue

        # IMM interaction
        imm.interaction()

        # Predict
        cv.predict()
        ct.predict()

        # CV update
        gated_cv = cv.gate(detections)
        logL_cv = cv.update(gated_cv)

        # CT update
        gated_ct = ct.gate(detections)
        logL_ct = ct.update(gated_ct)

        # IMM update
        imm.update_mode_probabilities(logL_cv, logL_ct)

        # Fusion
        imm.fuse()
        # ---------------- Logging ----------------

        cv.estimate_history.append(cv.state.state_vector.flatten().copy())
        cv.cov_trace_history.append(np.trace(cv.state.covar))
        cv.status_history.append(cv.status)

        ct.estimate_history.append(ct.state.state_vector.flatten().copy())
        ct.cov_trace_history.append(np.trace(ct.state.covar))
        ct.status_history.append(ct.status)

    return {
        "scene": scene,
        "cv_tracker": cv,
        "ct_tracker": ct,
        "imm_tracker": imm,
        "fused_history": np.array(imm.fused_history),
        "mode_probabilities": np.array(imm.mu_history)
    }
# ============================================================
# ======================= MAIN ===============================
# ============================================================

if __name__ == "__main__":

    for scene_type in ["aircraft", "stealth", "empty"]:

        scene = simulate_scene(scene_type, plot=False)

        #creating the object of CV and CT class. 
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

        for detections in scene["measurements"]["detections"]:

            # ---------------- Track Initialization ----------------
            if not cv.initialized:
                cv.step(detections)

            if not ct.initialized:
                ct.step(detections)

            if not (cv.initialized and ct.initialized):
                continue

            # ---------------- IMM Interaction ----------------
            imm.interaction()

            # ---------------- Predict ----------------
            cv.predict()
            ct.predict()

            # ---------------- CV Model ----------------
            gated_cv = cv.gate(detections)
            logL_cv = cv.update(gated_cv)
            #CT
            gated_ct = ct.gate(detections)
            logL_ct = ct.update(gated_ct)

            # ---------------- IMM Update ----------------
            imm.update_mode_probabilities(logL_cv, logL_ct)

            # ---------------- Fusion ----------------
            imm.fuse()

        # ==================================================
        # Convert histories
        # ==================================================

        fused = np.array(imm.fused_history)
        mu = np.array(imm.mu_history)

        # ==================================================
        # Plot 1 — Scene + Track
        # ==================================================

        plt.figure(figsize=(10,8))

        plt.plot(scene["truth"]["x"],
                 scene["truth"]["y"],
                 linewidth=2,
                 label="Truth")

        plt.scatter(scene["plot_data"]["clutter_x"],
                    scene["plot_data"]["clutter_y"],
                    s=6,
                    alpha=0.15,
                    label="Clutter")

        plt.scatter(scene["plot_data"]["detect_x"],
                    scene["plot_data"]["detect_y"],
                    s=20,
                    marker='x',
                    label="All Detections")

        if len(fused) > 0:
            plt.plot(fused[:,0],
                     fused[:,2],
                     color='green',
                     linewidth=2,
                     label="IMM (CV+CT) Track")

        plt.legend()
        plt.title(f"Radar Scene with IMM Track — {scene_type}")
        plt.grid(True)
        plt.show()

        # ==================================================
        # Plot 2 — Mode Probabilities
        # ==================================================

        if len(mu) > 0:
            plt.figure()
            plt.plot(mu[:,0], label="CV Probability")
            plt.plot(mu[:,1], label="CT Probability")
            plt.legend()
            plt.title(f"IMM Mode Probabilities — {scene_type}")
            plt.grid(True)
            plt.show()