import cv2
import numpy as np
import time


class LivenessDetector:
    def __init__(self, fps=30, blink_window=3.0, passive_window=1.0):

        self.fps = fps
        self.blink_window_frames = int(fps * blink_window)
        self.passive_window_frames = int(fps * passive_window)

        # ---- Blink ----
        self.ear_threshold = 0.25
        self.blink_frames_required = 1
        self.blink_counter = 0
        self.blink_detected = False
        self.frame_count = 0

        # ---- Passive buffers ----
        self.depth_buffer = []
        self.texture_buffer = []
        self.prev_gray = None

        self.stage = "WAIT_BLINK"
        self.decision_made = False
        self.live_result = None

    # -----------------------------------
    # EAR Calculation
    # -----------------------------------
    def compute_ear(self, lm, w, h, left=True):
        if left:
            ids = [33, 160, 158, 133, 153, 144]
        else:
            ids = [362, 385, 387, 263, 373, 380]

        pts = []
        for i in ids:
            x = lm.landmark[i].x * w
            y = lm.landmark[i].y * h
            pts.append((x, y))

        p1, p2, p3, p4, p5, p6 = pts

        A = np.linalg.norm(np.array(p2) - np.array(p6))
        B = np.linalg.norm(np.array(p3) - np.array(p5))
        C = np.linalg.norm(np.array(p1) - np.array(p4))

        return (A + B) / (2.0 * C + 1e-6)

    # -----------------------------------
    # Depth curvature
    # -----------------------------------
    def compute_depth(self, lm):
        indices = [33,133,159,145,153,154,263,362,386,374,380,381]
        z_vals = [lm.landmark[i].z for i in indices]
        return np.std(np.array(z_vals))

    # -----------------------------------
    # Texture variation
    # -----------------------------------
    def compute_texture(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        return np.mean(diff)

    # -----------------------------------
    # Main Check
    # -----------------------------------
    def check(self, frame, lm):

        if self.decision_made:
            return self.live_result

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1

        # =========================
        # STAGE 1: WAIT FOR BLINK
        # =========================
        if self.stage == "WAIT_BLINK":

            left_ear = self.compute_ear(lm, w, h, True)
            right_ear = self.compute_ear(lm, w, h, False)
            ear = (left_ear + right_ear) / 2.0

            if ear < self.ear_threshold:
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.blink_frames_required:
                    self.blink_detected = True
                self.blink_counter = 0

            # If blink detected → move to passive validation
            if self.blink_detected:
                self.stage = "PASSIVE_CHECK"
                self.frame_count = 0
                return None

            # If window expired without blink → FAIL
            if self.frame_count > self.blink_window_frames:
                self.live_result = False
                self.decision_made = True
                return False

            return None

        # =========================
        # STAGE 2: PASSIVE CHECK
        # =========================
        if self.stage == "PASSIVE_CHECK":

            depth_val = self.compute_depth(lm)
            texture_val = self.compute_texture(gray)

            self.depth_buffer.append(depth_val)
            self.texture_buffer.append(texture_val)

            if len(self.depth_buffer) > self.passive_window_frames:
                self.depth_buffer.pop(0)

            if len(self.texture_buffer) > self.passive_window_frames:
                self.texture_buffer.pop(0)

            # Wait until enough frames collected
            if len(self.depth_buffer) < self.passive_window_frames:
                return None

            depth_ok = np.mean(self.depth_buffer) > 0.0020
            texture_ok = np.mean(self.texture_buffer) > 40

            if depth_ok or texture_ok:
                self.live_result = True
            else:
                self.live_result = False

            self.decision_made = True
            return self.live_result

    # -----------------------------------
    # Reset
    # -----------------------------------
    def reset(self):
        self.__init__(self.fps)
