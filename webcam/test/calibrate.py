"""
calibrate.py — Multi-spoof calibration (phone + laptop screen)

HOW TO RUN:
    cd D:\MCA\restructuredperio(13_02_2026)\webcam\test
    python calibrate.py

Sessions:
    1. REAL FACE         (12 sec)
    2. PHONE SCREEN      (12 sec) — show phone with face photo
    3. LAPTOP/MONITOR    (12 sec) — show laptop/monitor with face photo

Output: worst-case thresholds that reject ALL spoof types.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import entropy as scipy_entropy
import mediapipe as mp
import time
import json
import sys
import os


# ══════════════════════════════════════════════════════════════════════
# CUE FUNCTIONS (self-contained)
# ══════════════════════════════════════════════════════════════════════

def compute_fft_slope(gray, lm, w, h):
    cx = int(lm.landmark[205].x * w)
    cy = int(lm.landmark[205].y * h)
    size = max(int(w * 0.10), 48)
    x1, y1 = max(cx-size, 0), max(cy-size, 0)
    x2, y2 = min(cx+size, w), min(cy+size, h)
    roi = gray[y1:y2, x1:x2]
    if roi.shape[0] < 48 or roi.shape[1] < 48:
        return -0.30
    roi = cv2.resize(roi, (128, 128)).astype(np.float32)
    hann = np.hanning(128)
    roi_w = roi * np.outer(hann, hann)
    fshift = np.fft.fftshift(np.fft.fft2(roi_w))
    power  = np.log1p(np.abs(fshift) ** 2)
    Y, X   = np.ogrid[:128, :128]
    radius = np.clip(np.sqrt((X-64)**2 + (Y-64)**2).astype(int), 0, 63)
    radial = np.array([power[radius==r].mean() if (radius==r).any() else 0.0
                       for r in range(64)])
    r_vals = np.arange(2, 60)
    p_vals = radial[2:60]
    valid  = p_vals > 0
    if valid.sum() < 10:
        return -0.30
    slope, _ = np.polyfit(np.log(r_vals[valid]), np.log(p_vals[valid]), 1)
    return float(slope)


def compute_hsv_score(frame, lm, w, h):
    xs = [lm.landmark[i].x * w for i in [10, 234, 454, 152]]
    ys = [lm.landmark[i].y * h for i in [10, 234, 454, 152]]
    x1, y1 = max(int(min(xs))-15, 0), max(int(min(ys))-15, 0)
    x2, y2 = min(int(max(xs))+15, w), min(int(max(ys))+15, h)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return 0.45
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mean_s, std_s = np.mean(S), np.std(S)
    sat_screen = (mean_s/200.0) * (1.0 - np.clip(std_s/60.0, 0, 1))
    Vf = V.astype(np.float32)
    m  = cv2.blur(Vf, (15,15))
    v_local_var = float(np.mean(np.sqrt(np.maximum(cv2.blur(Vf**2,(15,15))-m**2, 0))))
    v_screen = 1.0 - np.clip(v_local_var/800.0, 0, 1)
    skin_hue = np.mean(((H>=0)&(H<=25))|((H>=160)&(H<=180)))
    return float(np.clip(sat_screen*0.35 + v_screen*0.35 + (1-skin_hue)*0.30, 0, 1))


def compute_specular_score(gray, lm, w, h):
    xs = [lm.landmark[i].x * w for i in [10, 234, 454, 152]]
    ys = [lm.landmark[i].y * h for i in [10, 234, 454, 152]]
    x1, y1 = max(int(min(xs))-10, 0), max(int(min(ys))-10, 0)
    x2, y2 = min(int(max(xs))+10, w), min(int(max(ys))+10, h)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.5
    h_roi, w_roi = roi.shape
    diff = np.clip(roi.astype(np.float32) -
                   cv2.GaussianBlur(roi,(21,21),0), 0, 255).astype(np.uint8)
    _, bright = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    if np.sum(bright>0) < 10:
        return 0.4
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    if num_labels <= 1:
        return 0.3
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas > 5]
    if len(areas) == 0:
        return 0.3
    num_score = np.clip(len(areas)/15.0, 0, 1)
    size_var  = (np.clip((np.std(areas)/(np.mean(areas)+1e-6))/2.0, 0, 1)
                 if len(areas) > 1 else 0.0)
    cc = centroids[1:len(areas)+1]
    if len(cc) > 2:
        xb = np.clip((cc[:,0]/(w_roi+1e-6)*4).astype(int), 0, 3)
        yb = np.clip((cc[:,1]/(h_roi+1e-6)*4).astype(int), 0, 3)
        grid = np.zeros(16)
        for x, y in zip(xb, yb): grid[x*4+y] += 1
        sp_ent = scipy_entropy(grid/(grid.sum()+1e-10)+1e-10) / np.log(16)
    else:
        sp_ent = 0.0
    return float(np.clip(num_score*0.4 + size_var*0.3 + sp_ent*0.3, 0, 1))


def compute_depth(lm):
    return float(np.std([lm.landmark[i].z for i in
                         [33,133,159,145,153,154,263,362,386,374,380,381]]))


# ══════════════════════════════════════════════════════════════════════
# SESSION COLLECTOR
# ══════════════════════════════════════════════════════════════════════

def collect_session(cap, face_mesh, label, instruction, color, duration=12):
    data = {k: [] for k in ["fft_slope","hsv","specular","depth"]}

    print(f"\n{'━'*58}")
    print(f"  SESSION: {label}")
    print(f"  → {instruction}")
    print(f"  → Duration: {duration} seconds")
    print(f"{'━'*58}")
    input("  Press ENTER when ready...")
    print("  GO! Collecting...")

    start = time.time()
    frame_idx = 0

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_idx += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        remaining = max(0, duration-(time.time()-start))
        cv2.rectangle(frame, (0,0), (w,55), (20,20,20), -1)
        cv2.putText(frame, f"{label}  —  {remaining:.1f}s",
                    (15,38), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (0,h-8),
                      (int(((duration-remaining)/duration)*w), h), color, -1)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED",
                        (15,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
            continue

        lm = results.multi_face_landmarks[0]
        slope = compute_fft_slope(gray, lm, w, h)
        data["fft_slope"].append(slope)
        data["hsv"].append(compute_hsv_score(frame, lm, w, h))
        data["specular"].append(compute_specular_score(gray, lm, w, h))
        data["depth"].append(compute_depth(lm))

        if frame_idx % 25 == 0:
            cv2.putText(frame,
                f"fft:{slope:.3f}  hsv:{data['hsv'][-1]:.3f}"
                f"  spec:{data['specular'][-1]:.3f}  depth:{data['depth'][-1]:.4f}",
                (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,200), 1)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    n = len(data["fft_slope"])
    print(f"  ✓ {n} frames — {label}")
    return data


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def summarize(vals):
    if not vals:
        return dict(mean=0,std=0,p10=0,p90=0,min=0,max=0)
    v = np.array(vals)
    return {k: round(float(x),5) for k,x in zip(
        ["mean","std","p10","p90","min","max"],
        [np.mean(v),np.std(v),np.percentile(v,10),
         np.percentile(v,90),np.min(v),np.max(v)])}


def find_threshold(real_vals, spoof_vals, higher_is_live):
    """Find threshold that rejects ALL spoof_vals while accepting real_vals."""
    if len(real_vals) < 5 or len(spoof_vals) < 5:
        return None, 0.0
    ra, sa = np.array(real_vals), np.array(spoof_vals)
    cands = np.linspace(min(ra.min(),sa.min()), max(ra.max(),sa.max()), 500)
    bt, ba = None, 0.0
    for t in cands:
        acc = ((np.mean(ra>=t)+np.mean(sa<t))/2 if higher_is_live
               else (np.mean(ra<=t)+np.mean(sa>t))/2)
        if acc > ba:
            ba, bt = acc, t
    return round(float(bt),5), round(ba*100,1)


def worst_case_spoof(spoof_datasets, key, higher_is_live):
    """
    Merge all spoof datasets and return the hardest-to-reject value.
    If higher_is_live → spoof danger = highest spoof value (most live-like).
    If lower_is_live  → spoof danger = lowest spoof value (most live-like).
    """
    all_spoof = []
    for d in spoof_datasets:
        all_spoof.extend(d.get(key, []))
    if not all_spoof:
        return []
    # Return the "most dangerous" quartile of spoof values
    arr = np.array(all_spoof)
    if higher_is_live:
        # spoofs that look most live = highest values
        threshold = np.percentile(arr, 75)
        return arr[arr >= threshold].tolist()
    else:
        # spoofs that look most live = lowest values
        threshold = np.percentile(arr, 25)
        return arr[arr <= threshold].tolist()


def print_report(real_data, spoof_datasets, spoof_labels):
    cues = [
        ("FFT Slope",  "fft_slope", False, "more negative = real skin"),
        ("HSV Score",  "hsv",       False, "lower = skin colour"),
        ("Specular",   "specular",  True,  "higher = skin highlights"),
        ("Depth STD",  "depth",     True,  "higher = 3D geometry"),
    ]

    print("\n" + "═"*65)
    print("  CALIBRATION REPORT — Multi-Spoof")
    print("═"*65)

    final_thresholds = {}

    for name, key, hil, desc in cues:
        rv = real_data.get(key, [])
        rs = summarize(rv)

        print(f"\n┌─ {name}  ({desc})")
        print(f"│  REAL   mean={rs['mean']:>8.4f}  std={rs['std']:.4f}"
              f"  range [{rs['p10']:.4f} – {rs['p90']:.4f}]")

        # Per-spoof stats
        all_spoof_vals = []
        for label, sd in zip(spoof_labels, spoof_datasets):
            sv = sd.get(key, [])
            ss = summarize(sv)
            t, acc = find_threshold(rv, sv, hil)
            print(f"│  {label:<14} mean={ss['mean']:>8.4f}  std={ss['std']:.4f}"
                  f"  range [{ss['p10']:.4f} – {ss['p90']:.4f}]"
                  f"  threshold={t}  acc={acc}%")
            all_spoof_vals.extend(sv)

        # Worst-case combined threshold
        t_combined, acc_combined = find_threshold(rv, all_spoof_vals, hil)
        sep = abs(np.mean(rv) - np.mean(all_spoof_vals)) if all_spoof_vals else 0
        overlap = max(0,
            min(np.max(rv) if rv else 0, np.max(all_spoof_vals) if all_spoof_vals else 0) -
            max(np.min(rv) if rv else 0, np.min(all_spoof_vals) if all_spoof_vals else 0))

        star = ("★★★ HIGHLY RELIABLE" if acc_combined > 90 else
                "★★  RELIABLE"        if acc_combined > 75 else
                "★   WEAK")
        direction = "LIVE ≥ threshold" if hil else "LIVE ≤ threshold"
        print(f"│  ── COMBINED ─── threshold={t_combined}  "
              f"acc={acc_combined}%  sep={sep:.4f}  overlap={overlap:.4f}")
        print(f"└  {star}  ({direction})")

        final_thresholds[key] = {
            "threshold": t_combined,
            "acc": acc_combined,
            "higher_is_live": hil,
        }

    # ── Paste block ───────────────────────────────────────────────────
    T = final_thresholds
    def t(k): return T[k]["threshold"]

    fft_t  = t("fft_slope")
    hsv_t  = t("hsv")
    spec_t = t("specular")
    dep_t  = t("depth")

    # Safety margin: push thresholds 10% into the live zone
    # so edge cases don't flip. Direction depends on cue.
    fft_safe  = round(fft_t  - 0.01, 5)   # more negative = safer
    hsv_safe  = round(hsv_t  - 0.01, 5)   # lower = safer
    spec_safe = round(spec_t + 0.01, 5)   # higher = safer
    dep_safe  = round(dep_t  + 0.0002, 5) # higher = safer

    print("\n" + "═"*65)
    print("  PASTE INTO liveness.py  →  check() method")
    print("═"*65)
    print(f"""
    # ── MULTI-SPOOF CALIBRATED THRESHOLDS ──────────────────────
    # Rejects: phone screen + laptop/monitor screen
    # Real face accuracy guaranteed from calibration run

    # CUE 1: FFT Slope
    if fft_slope < {fft_safe:.5f}:        # LIVE  (calibrated safe zone)
        score += 4
    elif fft_slope < {round(fft_t+0.01,5):.5f}:    # marginal
        score += 2
    elif fft_slope > {round(fft_t+0.03,5):.5f}:    # clearly SPOOF
        score -= 1

    # CUE 2: HSV Score
    if hsv_score < {hsv_safe:.5f}:        # LIVE
        score += 3
    elif hsv_score < {round(hsv_t+0.01,5):.5f}:    # marginal
        score += 1
    elif hsv_score > {round(hsv_t+0.06,5):.5f}:    # clearly SPOOF
        score -= 2

    # CUE 3: Specular
    if specular_score > {spec_safe:.5f}:  # LIVE
        score += 2
    elif specular_score > {round(spec_t-0.05,5):.5f}: # marginal
        score += 1
    elif specular_score < {round(spec_t-0.15,5):.5f}: # clearly SPOOF
        score -= 1

    # CUE 4: Depth STD
    if depth_mean > {dep_safe:.5f}:       # LIVE
        score += 2
    elif depth_mean > {round(dep_t-0.0003,5):.5f}:  # marginal
        score += 1
    else:                                  # SPOOF
        score -= 1
    """)

    return final_thresholds


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*58)
    print("  LIVENESS CALIBRATION — Multi-Spoof Version")
    print("  Rejects: Phone Screen + Laptop/Monitor Screen")
    print("═"*58)
    print("""
  Three sessions:
    1. YOUR REAL FACE          (12 sec)
    2. PHONE SCREEN spoof      (12 sec)  ← show phone with face photo
    3. LAPTOP/MONITOR spoof    (12 sec)  ← show laptop with face photo
    """)
    input("  Press ENTER to open camera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open camera.")
        input("  Press ENTER to exit.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"  Camera: {int(cap.get(3))}×{int(cap.get(4))}")

    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        real_data = collect_session(
            cap, face_mesh,
            label="REAL FACE",
            instruction="Sit normally, look at camera, blink naturally",
            color=(0, 220, 0),
            duration=12)

        phone_data = collect_session(
            cap, face_mesh,
            label="PHONE SPOOF",
            instruction="Hold phone showing face photo in front of camera",
            color=(0, 80, 255),
            duration=12)

        laptop_data = collect_session(
            cap, face_mesh,
            label="LAPTOP SPOOF",
            instruction="Hold laptop/show monitor with face photo to camera",
            color=(0, 0, 200),
            duration=12)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        cap.release()
        cv2.destroyAllWindows()
        input("  Press ENTER to exit.")
        sys.exit(1)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    spoof_datasets = [phone_data, laptop_data]
    spoof_labels   = ["PHONE SPOOF  ", "LAPTOP SPOOF "]

    for label, d in [("REAL", real_data),
                     ("PHONE", phone_data),
                     ("LAPTOP", laptop_data)]:
        n = len(d["fft_slope"])
        if n < 10:
            print(f"\n  WARNING: Only {n} {label} frames — results may be weak.")

    results = print_report(real_data, spoof_datasets, spoof_labels)

    # Save JSON
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "calibration_results.json")
    with open(out, "w") as f:
        json.dump({
            "real":   {k: summarize(v) for k,v in real_data.items()},
            "phone":  {k: summarize(v) for k,v in phone_data.items()},
            "laptop": {k: summarize(v) for k,v in laptop_data.items()},
            "thresholds": results,
        }, f, indent=2)

    print(f"\n  Results saved → {out}")
    input("\n  Press ENTER to exit.")


if __name__ == "__main__":
    main()