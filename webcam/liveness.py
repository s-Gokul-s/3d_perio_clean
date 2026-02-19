import cv2
import numpy as np


class LivenessDetector:

    def __init__(self, fps=30, window_seconds=2.5):

        self.fps = fps
        self.window_frames = int(fps * window_seconds)

        # Buffers
        self.depth_buffer = []
        self.motion_buffer = []

        self.landmark_history = []
        self.prev_gray = None

        # Blink
        self.ear_threshold = 0.23
        self.blink_counter = 0
        self.blink_detected = False
        self.blink_frames_required = 1

        # State
        self.frame_count = 0
        self.decision_made = False
        self.live_result = None

    # -----------------------------------------
    # EAR (Blink Detection)
    # -----------------------------------------
    def compute_ear(self, lm, w, h, left=True):

        ids = [33,160,158,133,153,144] if left else \
              [362,385,387,263,373,380]

        pts = [(lm.landmark[i].x * w,
                lm.landmark[i].y * h) for i in ids]

        p1,p2,p3,p4,p5,p6 = pts

        A = np.linalg.norm(np.array(p2)-np.array(p6))
        B = np.linalg.norm(np.array(p3)-np.array(p5))
        C = np.linalg.norm(np.array(p1)-np.array(p4))

        return (A+B)/(2.0*C + 1e-6)

    # -----------------------------------------
    # Depth Curvature
    # -----------------------------------------
    def compute_depth(self, lm):

        indices = [33,133,159,145,153,154,
                   263,362,386,374,380,381]

        z_vals = [lm.landmark[i].z for i in indices]

        return np.std(np.array(z_vals))

    # -----------------------------------------
    # Motion (Organic vs Rigid)
    # -----------------------------------------
    def compute_motion(self, lm):

        key_points = [33,133,263,362]

        coords = np.array([
            [lm.landmark[i].x,
            lm.landmark[i].y]
            for i in key_points
        ])

        self.landmark_history.append(coords)

        if len(self.landmark_history) < 2:
            return 0, False

        if len(self.landmark_history) > self.window_frames:
            self.landmark_history.pop(0)

        motion = np.diff(self.landmark_history, axis=0)
        motion_mag = np.linalg.norm(motion, axis=2)

        mean_motion = np.mean(motion_mag)
        std_motion = np.std(motion_mag)

        rigidity_ratio = std_motion / (mean_motion + 1e-6)

        rigid_motion = rigidity_ratio < 0.15

        return mean_motion, rigid_motion


    # -----------------------------------------
    # MAIN CHECK
    # -----------------------------------------
    def check(self, frame, lm):

        if self.decision_made:
            return self.live_result

        h, w = frame.shape[:2]

        self.frame_count += 1

        # ---- Blink ----
        left_ear = self.compute_ear(lm, w, h, True)
        right_ear = self.compute_ear(lm, w, h, False)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.ear_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_frames_required:
                self.blink_detected = True
            self.blink_counter = 0

        # ---- Depth ----
        depth_val = self.compute_depth(lm)
        self.depth_buffer.append(depth_val)

        # ---- Motion ----
        motion_val, rigid_motion = self.compute_motion(lm)
        self.motion_buffer.append(motion_val)

        # Maintain window
        if len(self.depth_buffer) > self.window_frames:
            self.depth_buffer.pop(0)

        if len(self.motion_buffer) > self.window_frames:
            self.motion_buffer.pop(0)

        # Wait full window
        if self.frame_count < self.window_frames:
            return None

        # --------------------------------
        # CALIBRATED FUSION (Based on your data)
        # --------------------------------

        depth_mean = np.mean(self.depth_buffer)
        motion_mean = np.mean(self.motion_buffer)

        print("---- LIVENESS DEBUG ----")
        print("Depth:", depth_mean)
        print("Motion:", motion_mean)
        print("Blink:", self.blink_detected)
        print("Rigid:", rigid_motion)
        print("------------------------")

        # 🔥 Thresholds tuned using YOUR measurements
        depth_ok = depth_mean > 0.004        # Real ≈ 0.0055
        motion_range_ok = 3 < motion_mean < 12
        organic_motion = not rigid_motion

        # Strong fusion logic
        if self.blink_detected:
            live = True
        elif depth_ok and motion_range_ok and organic_motion:
            live = True
        else:
            live = False

        self.live_result = live
        self.decision_made = True

        return self.live_result

    # -----------------------------------------
    # Reset
    # -----------------------------------------
    def reset(self):
        self.__init__(self.fps)
