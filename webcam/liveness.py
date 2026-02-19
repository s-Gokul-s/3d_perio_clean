import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import entropy as scipy_entropy


class LivenessDetector:
    """
    Calibrated for Samsung S22 + LED indoor lighting.
    Rejects: phone screen (AMOLED) + laptop screen.

    Cue reliability from calibration data:
      HSV Score  : real=0.432  phone=0.463  laptop=0.456  acc=97.6%  ✅ KEEP
      Depth STD  : real=0.0075 phone=0.0031 laptop=0.0036 acc=100%   ✅ KEEP
      FFT Slope  : real=-0.347 phone=-0.343 OVERLAP                  ❌ DROPPED
      Specular   : laptop=0.719 overlaps real=0.758                  ❌ DROPPED as primary

    New cues added to compensate:
      Reflection variance  — screens have unnaturally stable brightness
      Chroma noise         — real skin has micro colour noise; screens don't
      rPPG                 — blood flow signal; screens have none
    """

    def __init__(self, fps=30, window_seconds=5.0):
        self.fps = fps
        self.window_frames = int(fps * window_seconds)

        # Buffers
        self.hsv_buffer        = []
        self.depth_buffer      = []
        self.ref_var_buffer    = []   # reflection variance
        self.chroma_buffer     = []   # chroma noise
        self.rppg_g            = []

        # Blink
        self.ear_history       = []
        self.blink_counter     = 0
        self.confirmed_blinks  = 0
        self.was_open          = True
        self.prev_gray         = None

        self.frame_count   = 0
        self.decision_made = False
        self.live_result   = None

    # ================================================================
    # CUE 1 — HSV SCORE  ✅ acc=97.6%
    #
    # Calibrated thresholds:
    #   real   mean=0.432  p90=0.442
    #   phone  mean=0.463  p10=0.454
    #   laptop mean=0.456  p10=0.452
    #   Combined threshold = 0.443  (clean gap between real p90 and spoof p10)
    # ================================================================
    def compute_hsv_score(self, frame, lm, w, h):
        xs = [lm.landmark[i].x * w for i in [10, 234, 454, 152]]
        ys = [lm.landmark[i].y * h for i in [10, 234, 454, 152]]
        x1 = max(int(min(xs))-15, 0);  y1 = max(int(min(ys))-15, 0)
        x2 = min(int(max(xs))+15, w);  y2 = min(int(max(ys))+15, h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return 0.45

        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        mean_s, std_s = np.mean(S), np.std(S)
        sat_screen = (mean_s/200.0) * (1.0 - np.clip(std_s/60.0, 0, 1))

        Vf = V.astype(np.float32)
        m  = cv2.blur(Vf, (15,15))
        v_local_var = float(np.mean(
            np.sqrt(np.maximum(cv2.blur(Vf**2,(15,15)) - m**2, 0))))
        v_screen = 1.0 - np.clip(v_local_var/800.0, 0, 1)

        skin_hue = np.mean(((H>=0)&(H<=25))|((H>=160)&(H<=180)))
        return float(np.clip(
            sat_screen*0.35 + v_screen*0.35 + (1-skin_hue)*0.30, 0, 1))

    # ================================================================
    # CUE 2 — DEPTH STD  ✅ acc=100%
    #
    # Calibrated thresholds:
    #   real   mean=0.0075  p10=0.0060
    #   phone  mean=0.0031  p90=0.0033
    #   laptop mean=0.0036  p90=0.0036
    #   Safe live threshold = 0.0040 (above both spoof p90 values)
    # ================================================================
    def compute_depth(self, lm):
        indices = [33,133,159,145,153,154,263,362,386,374,380,381]
        return float(np.std([lm.landmark[i].z for i in indices]))

    # ================================================================
    # CUE 3 — REFLECTION VARIANCE (new)
    #
    # Screens have a backlight → brightness of any face region is driven
    # by the display, which is stable frame-to-frame.
    # Real faces under LED have micro-fluctuations from breathing,
    # blood flow, micro-movement → brightness varies slightly over time.
    #
    # Measure: std of mean brightness of forehead patch over the window.
    # Real face: higher temporal std.
    # Screen:    very low temporal std (stable backlight).
    # ================================================================
    def compute_reflection_variance(self, frame, lm, w, h):
        """Returns instantaneous forehead brightness — we track its std over time."""
        cx = int(lm.landmark[10].x * w)
        cy = int(lm.landmark[10].y * h)
        size = max(int(w * 0.07), 16)
        x1 = max(cx-size, 0);  y1 = max(cy-size, 0)
        x2 = min(cx+size, w);  y2 = min(cy+size, h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(roi.mean())   # stored per-frame; std computed at decision time

    # ================================================================
    # CUE 4 — CHROMA NOISE (new)
    #
    # Real skin has micro-level colour variation from blood vessels,
    # melanin distribution, pores — visible as high-frequency noise
    # in the chroma channels of YCrCb.
    #
    # Screens rendering a photo: chroma is quantised to display gamut,
    # compressed by JPEG, then re-encoded → much lower high-freq chroma noise.
    #
    # Measure: mean absolute difference between Cr channel and its
    # Gaussian-blurred version (high-pass chroma energy).
    # Real face: higher chroma noise.
    # Screen:    lower (smoother chroma).
    # ================================================================
    def compute_chroma_noise(self, frame, lm, w, h):
        xs = [lm.landmark[i].x * w for i in [234, 454]]  # cheeks
        ys = [lm.landmark[i].y * h for i in [234, 454]]
        cx = int(np.mean(xs));  cy = int(np.mean(ys))
        size = max(int(w * 0.12), 40)
        x1 = max(cx-size, 0);  y1 = max(cy-size, 0)
        x2 = min(cx+size, w);  y2 = min(cy+size, h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        cr = ycrcb[:,:,1]

        # High-pass: original minus blurred
        cr_blur = cv2.GaussianBlur(cr, (9,9), 0)
        hp = np.abs(cr - cr_blur)
        return float(np.mean(hp))

    # ================================================================
    # CUE 5 — rPPG (supporting)
    # ================================================================
    def extract_rppg(self, frame, lm, w, h):
        size = max(int(w * 0.055), 14)
        vals = []
        for idx in [10, 151, 205, 425]:
            cx = int(lm.landmark[idx].x * w)
            cy = int(lm.landmark[idx].y * h)
            roi = frame[max(cy-size,0):min(cy+size,h),
                        max(cx-size,0):min(cx+size,w)]
            if roi.size > 0:
                vals.append(float(roi[:,:,1].mean()))
        if vals:
            self.rppg_g.append(np.mean(vals))

    def compute_rppg_score(self):
        if len(self.rppg_g) < self.window_frames:
            return 0.0
        signal = np.array(self.rppg_g[-self.window_frames:])
        x = np.arange(len(signal))
        signal = signal - np.polyval(np.polyfit(x, signal, 1), x)
        if np.std(signal) < 1e-6:
            return 0.0
        signal = signal / (np.std(signal) + 1e-8)
        nyq = self.fps / 2.0
        try:
            b, a = butter(4, [0.7/nyq, min(3.5/nyq, 0.98)], btype='band')
            filtered = filtfilt(b, a, signal)
        except Exception:
            return 0.0
        snr = float(np.var(filtered) / (np.var(signal) + 1e-10))
        freqs   = np.fft.rfftfreq(len(filtered), d=1.0/self.fps)
        fft_mag = np.abs(np.fft.rfft(filtered))
        if len(fft_mag) > 1:
            dominant = freqs[np.argmax(fft_mag[1:]) + 1]
            if not (0.6 <= dominant <= 4.0):
                return snr * 0.2
        return snr

    # ================================================================
    # EAR / BLINK
    # ================================================================
    def compute_ear(self, lm, w, h, left=True):
        ids = ([33,160,158,133,153,144] if left
               else [362,385,387,263,373,380])
        pts = [(lm.landmark[i].x*w, lm.landmark[i].y*h) for i in ids]
        p1,p2,p3,p4,p5,p6 = pts
        A = np.linalg.norm(np.array(p2)-np.array(p6))
        B = np.linalg.norm(np.array(p3)-np.array(p5))
        C = np.linalg.norm(np.array(p1)-np.array(p4))
        return (A+B) / (2.0*C + 1e-6)

    def update_blink(self, ear):
        EAR_OPEN, EAR_CLOSED, MAX_F = 0.28, 0.21, 5
        if ear < EAR_CLOSED:
            self.blink_counter += 1
            self.was_open = False
        elif ear > EAR_OPEN:
            if 1 <= self.blink_counter <= MAX_F:
                self.confirmed_blinks += 1
            self.blink_counter = 0
            self.was_open = True
        self.ear_history.append(ear)

    # ================================================================
    # MAIN CHECK
    # ================================================================
    def check(self, frame, lm):
        if self.decision_made:
            return self.live_result

        h, w = frame.shape[:2]
        self.frame_count += 1

        # Collect every frame
        ear = (self.compute_ear(lm, w, h, True) +
               self.compute_ear(lm, w, h, False)) / 2.0
        self.update_blink(ear)

        self.extract_rppg(frame, lm, w, h)

        self.hsv_buffer.append(self.compute_hsv_score(frame, lm, w, h))
        self.depth_buffer.append(self.compute_depth(lm))
        self.ref_var_buffer.append(self.compute_reflection_variance(frame, lm, w, h))
        self.chroma_buffer.append(self.compute_chroma_noise(frame, lm, w, h))

        for buf in [self.hsv_buffer, self.depth_buffer, self.ref_var_buffer,
                    self.chroma_buffer, self.rppg_g, self.ear_history]:
            if len(buf) > self.window_frames:
                buf.pop(0)

        if self.frame_count < self.window_frames:
            return None

        # ── Aggregate ──────────────────────────────────────────────
        hsv_score   = float(np.mean(self.hsv_buffer))
        depth_mean  = float(np.mean(self.depth_buffer))
        rppg_snr    = self.compute_rppg_score()

        # Reflection variance: std of brightness over time window
        # Real face: > ~1.5  |  Screen: < ~0.8  (needs your calibration)
        ref_var     = float(np.std(self.ref_var_buffer))

        # Chroma noise: mean high-freq chroma energy
        # Real face: > ~2.0  |  Screen: < ~1.2  (needs your calibration)
        chroma_noise = float(np.mean(self.chroma_buffer))

        # ── Debug ──────────────────────────────────────────────────
        print("\n==== LIVENESS DEBUG ====")
        print(f"HSV score:      {hsv_score:.5f}  "
              f"LIVE<0.443 | SPOOF>0.452  → "
              f"{'LIVE' if hsv_score<0.443 else 'SPOOF'}")
        print(f"Depth STD:      {depth_mean:.5f}  "
              f"LIVE>0.004 | SPOOF<0.0036 → "
              f"{'LIVE' if depth_mean>0.004 else 'SPOOF'}")
        print(f"Reflect var:    {ref_var:.4f}   "
              f"LIVE>1.5   | SPOOF<0.8   → "
              f"{'LIVE' if ref_var>1.5 else 'SPOOF (est.)'}")
        print(f"Chroma noise:   {chroma_noise:.4f}   "
              f"LIVE>2.0   | SPOOF<1.2   → "
              f"{'LIVE' if chroma_noise>2.0 else 'SPOOF (est.)'}")
        print(f"rPPG SNR:       {rppg_snr:.5f}  (supporting)")
        print(f"Blinks:         {self.confirmed_blinks}          (supporting)")
        print("========================")

        # ── FUSION ─────────────────────────────────────────────────
        score   = 0
        reasons = []

        # CUE 1: HSV  — calibrated, reliable  (max ±3)
        if hsv_score < 0.443:            # real p90=0.442 → clearly live
            score += 3
            reasons.append("hsv(+3)")
        elif hsv_score < 0.452:          # marginal zone
            score += 1
            reasons.append("hsv(+1)")
        elif hsv_score > 0.464:          # phone p10=0.454, laptop p10=0.452
            score -= 2
            reasons.append("hsv(-2)_spoof")
        else:
            reasons.append("hsv(0)")

        # CUE 2: Depth  — calibrated, 100% reliable  (max ±3)
        if depth_mean > 0.0040:          # above both spoof p90 values
            score += 3
            reasons.append("depth(+3)")
        elif depth_mean > 0.0036:        # marginal (above laptop p90)
            score += 1
            reasons.append("depth(+1)")
        else:                            # at or below spoof range
            score -= 2
            reasons.append("depth(-2)_spoof")

        # CUE 3: Reflection variance  — needs your calibration values
        # These are estimates — watch the debug output to tune them
        if ref_var > 1.5:
            score += 2
            reasons.append("refvar(+2)")
        elif ref_var > 0.8:
            score += 1
            reasons.append("refvar(+1)")
        else:
            score -= 1
            reasons.append("refvar(-1)_screen")

        # CUE 4: Chroma noise  — needs your calibration values
        if chroma_noise > 2.0:
            score += 2
            reasons.append("chroma(+2)")
        elif chroma_noise > 1.2:
            score += 1
            reasons.append("chroma(+1)")
        else:
            score -= 1
            reasons.append("chroma(-1)_screen")

        # CUE 5: rPPG (soft bonus)
        if rppg_snr > 0.05:
            score += 1
            reasons.append("rppg(+1)")

        # CUE 6: Blink (soft bonus)
        if self.confirmed_blinks >= 1:
            score += 1
            reasons.append("blink(+1)")

        # ── Decision ───────────────────────────────────────────────
        # Max possible: 3+3+2+2+1+1 = 12
        # Real face minimum (HSV+Depth alone): 3+3 = 6
        # Spoof maximum (HSV+Depth penalties): -2-2 = -4
        THRESHOLD = 6
        self.live_result   = (score >= THRESHOLD)
        self.decision_made = True

        print(f"Score: {score}/12   threshold={THRESHOLD}")
        print(f"Cues:  {reasons}")
        print(f"Decision: {'✓ LIVE' if self.live_result else '✗ SPOOF'}\n")

        return self.live_result

    def reset(self):
        self.__init__(self.fps)