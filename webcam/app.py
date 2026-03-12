import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import torch
import time
from PIL import Image

from model_setup import load_model, get_transform
from feature_extraction import get_signature
from database import load_database, save_database
from authentication import authenticate
from config import *
from enrollment import handle_enrollment
from preprocessing import enhance_crop
from liveness import LivenessDetector

st.set_page_config(
    page_title="PerioGuard — Biometric System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Playfair+Display:wght@700;800&display=swap');

*, html, body { box-sizing:border-box; margin:0; padding:0; }

/* ── ROOT PALETTE ─────────────────────────────── */
:root {
    --bg:        #f0f2f5;
    --surface:   #ffffff;
    --surface2:  #f8f9fb;
    --border:    #e2e6ed;
    --border2:   #d0d6e0;
    --text:      #0f1117;
    --text2:     #4a5568;
    --text3:     #8a96a8;
    --indigo:    #4f46e5;
    --indigo2:   #6366f1;
    --indigo-bg: #eef0ff;
    --green:     #059669;
    --green-bg:  #ecfdf5;
    --red:       #dc2626;
    --red-bg:    #fef2f2;
    --amber:     #d97706;
    --amber-bg:  #fffbeb;
    --blue:      #2563eb;
    --blue-bg:   #eff6ff;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}
/* Make the top header bar transparent so it blends with background */
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* ── SIDEBAR ──────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

/* ── MAIN BLOCK ───────────────────────────────── */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding: 0 !important; margin: 0 !important; max-width: 100% !important;
}
/* Push content below Streamlit's top header bar so title is never obscured */
[data-testid="stMain"] > div:first-child {
    padding-top: 3.2rem !important;
}

/* ── BUTTONS — reset all first ────────────────── */
.stButton > button {
    background: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    color: var(--text2) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: var(--indigo) !important;
    color: var(--indigo) !important;
    background: var(--indigo-bg) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.12) !important;
}

/* ── TEXT INPUT ───────────────────────────────── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.83rem !important;
    padding: 9px 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
    outline: none !important;
}
.stTextInput label { color: var(--text2) !important; font-size: 0.75rem !important; font-family:'Syne',sans-serif !important; }
.stCheckbox label  { color: var(--text2) !important; font-size: 0.8rem !important; font-family:'Syne',sans-serif !important; }
.stCheckbox [data-testid="stCheckbox"] input:checked + div {
    background: var(--indigo) !important;
    border-color: var(--indigo) !important;
}

/* ── STATUS BAR ───────────────────────────────── */
.sb {
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 18px; border-radius:10px;
    font-family:'JetBrains Mono',monospace;
    font-size:0.82rem; font-weight:500; border:1.5px solid;
    backdrop-filter: blur(8px);
}
.sb-granted { background:var(--green-bg);  border-color:#a7f3d0; color:var(--green); }
.sb-denied  { background:var(--red-bg);    border-color:#fecaca; color:var(--red);   }
.sb-spoof   { background:var(--amber-bg);  border-color:#fde68a; color:var(--amber); }
.sb-scan    { background:var(--blue-bg);   border-color:#bfdbfe; color:var(--blue);  }
.sb-idle    { background:var(--blue-bg);   border-color:#bfdbfe;       color:var(--blue);  }
.sb-right   { display:flex; align-items:center; gap:14px; }
.sb-score   { font-size:1.6rem; font-weight:700; }

/* ── ENROLL BADGES ────────────────────────────── */
.ep-badge { display:inline-block; font-size:0.68rem; font-weight:700;
            padding:4px 13px; border-radius:99px; margin-bottom:5px;
            font-family:'JetBrains Mono',monospace; letter-spacing:0.03em; }
.ep-p1   { background:var(--indigo-bg); color:var(--indigo);  border:1px solid #c7d2fe; }
.ep-p2   { background:var(--green-bg);  color:var(--green);   border:1px solid #a7f3d0; }
.ep-paus { background:var(--amber-bg);  color:var(--amber);   border:1px solid #fde68a; }
.ep-done { background:var(--green-bg);  color:var(--green);   border:1px solid #a7f3d0; }
.ep-track { background:var(--border); border-radius:99px; height:3px; overflow:hidden; margin:6px 0; }
.ep-fill  { height:100%; border-radius:99px;
            background:linear-gradient(90deg,var(--indigo),var(--indigo2)); transition:width .25s; }

/* ── DB ROWS ──────────────────────────────────── */
.db-row { display:flex; justify-content:space-between; align-items:center;
          padding:8px 12px; background:var(--surface2); border:1.5px solid var(--border);
          border-radius:8px; margin-bottom:5px; }
.db-name { font-size:0.85rem; font-weight:600; color:var(--text); font-family:'Syne',sans-serif; }
.db-tpl  { font-size:0.6rem; color:var(--text3); font-family:'JetBrains Mono',monospace;
           background:var(--border); padding:2px 8px; border-radius:99px; }

/* ── CARD ─────────────────────────────────────── */
.card {
    background:var(--surface); border:1.5px solid var(--border);
    border-radius:12px; padding:16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "initialized" not in st.session_state:
    st.session_state.model             = load_model()
    st.session_state.transform         = get_transform()
    st.session_state.db                = load_database()
    st.session_state.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state.model.eval()
    st.session_state.face_mesh         = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    st.session_state.liveness_detector = LivenessDetector()
    st.session_state.score_history     = []
    st.session_state.decision_locked   = False
    st.session_state.locked_result     = ""
    st.session_state.locked_score      = 0.0
    st.session_state.attack_on         = False
    st.session_state.epsilon           = 0.005
    st.session_state.auth_active       = False
    st.session_state.liveness_passed   = False
    st.session_state.page              = "AUTH"
    st.session_state.initialized       = True

# ══════════════════════════════════════════════════════════════════
# FGSM — identical to main.py
# ══════════════════════════════════════════════════════════════════
def fgsm_attack(model, tensor, eps):
    tensor = tensor.clone().detach().to(st.session_state.device)
    tensor.requires_grad_(True)
    logits = model(tensor).logits
    label  = torch.argmax(logits, dim=1)
    torch.nn.CrossEntropyLoss()(logits, label).backward()
    return torch.clamp(tensor + eps * tensor.grad.sign(), -1, 1).detach()

def get_sig_from_tensor(tensor):
    with torch.no_grad():
        out = st.session_state.model.vit(tensor)
    return out.last_hidden_state[0, 0].cpu().numpy()

def reset_auth():
    st.session_state.decision_locked   = False
    st.session_state.locked_result     = ""
    st.session_state.locked_score      = 0.0
    st.session_state.score_history.clear()
    st.session_state.liveness_passed   = False
    st.session_state.auth_active       = False
    try:
        st.session_state.liveness_detector.reset()
    except AttributeError:
        st.session_state.liveness_detector = LivenessDetector()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    n_users = len(st.session_state.db)

    # Brand header
    st.markdown("""
    <div style='padding:24px 18px 0'>
        <div style='display:flex;align-items:center;gap:11px;margin-bottom:20px'>
            <div style='width:36px;height:36px;border-radius:10px;
                        background:linear-gradient(135deg,#4f46e5,#6366f1);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem;box-shadow:0 4px 12px rgba(79,70,229,0.3)'>👁</div>
            <div>
                <div style='font-size:1.25rem;font-weight:800;color:#0f1117;
                            font-family:"Playfair Display",serif;letter-spacing:-0.02em'>PerioGuard</div>
                <div style='font-size:.58rem;color:#8a96a8;font-family:"JetBrains Mono",monospace;
                            letter-spacing:2.5px;text-transform:uppercase;margin-top:1px'>Biometric System</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav label
    st.markdown("""
    <div style='padding:0 18px;margin-bottom:6px'>
        <span style='font-size:.58rem;font-weight:700;letter-spacing:2.5px;
                     text-transform:uppercase;color:#8a96a8;font-family:"JetBrains Mono",monospace'>
            Navigation
        </span>
    </div>
    <div style='padding:0 10px'>
    """, unsafe_allow_html=True)

    if st.button("🔐  Authentication", key="nav_auth", use_container_width=True):
        st.session_state.page = "AUTH"
        reset_auth()
        st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("📋  Enrollment", key="nav_enroll", use_container_width=True):
        st.session_state.page = "ENROLL"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Status panel
    db_color  = "#059669" if n_users > 0 else "#dc2626"
    db_label  = "READY"   if n_users > 0 else "EMPTY"
    st.markdown(f"""
    <div style='margin:20px 10px 0;padding:14px 16px;background:#f8f9fb;
                border:1.5px solid #e2e6ed;border-radius:10px;'>
        <div style='font-size:.56rem;font-weight:700;letter-spacing:2px;
                    text-transform:uppercase;color:#8a96a8;
                    font-family:"JetBrains Mono",monospace;margin-bottom:10px'>
            System Status
        </div>
        <div style='display:grid;gap:7px'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:.8rem;color:#4a5568;font-family:"Syne",sans-serif'>Database</span>
                <span style='font-size:.65rem;font-family:"JetBrains Mono",monospace;
                             font-weight:600;color:{db_color};
                             background:{"#ecfdf5" if n_users>0 else "#fef2f2"};
                             padding:2px 8px;border-radius:99px'>● {db_label}</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:.8rem;color:#4a5568;font-family:"Syne",sans-serif'>Users</span>
                <span style='font-size:.65rem;font-family:"JetBrains Mono",monospace;
                             font-weight:600;color:#4f46e5'>{n_users} enrolled</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:.8rem;color:#4a5568;font-family:"Syne",sans-serif'>Model</span>
                <span style='font-size:.65rem;font-family:"JetBrains Mono",monospace;
                             font-weight:600;color:#059669;background:#ecfdf5;
                             padding:2px 8px;border-radius:99px'>● LOADED</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CAMERA
# ══════════════════════════════════════════════════════════════════
if "cap" not in st.session_state or not st.session_state.cap.isOpened():
    if "cap" in st.session_state:
        st.session_state.cap.release()
    _cap = cv2.VideoCapture(0)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    st.session_state.cap = _cap
cap = st.session_state.cap

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — AUTHENTICATION
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "AUTH":

    st.markdown("""
    <div style='padding:22px 28px 12px;border-bottom:1.5px solid #e2e6ed;
                background:#fff;margin-bottom:0'>
        <div style='display:flex;align-items:baseline;gap:14px'>
            <span style='font-size:1.45rem;font-weight:800;color:#0f1117;
                         font-family:"Syne",sans-serif;letter-spacing:-0.02em'>Authentication</span>
            <span style='font-size:.58rem;color:#8a96a8;font-family:"JetBrains Mono",monospace;
                         letter-spacing:2.5px;text-transform:uppercase'>
                Periocular Identity Verification
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout ─────────────────────────────────────────────────
    _l, cam_col, right_col, _r = st.columns([0.5, 4, 2.5, 0.3], gap="small")

    with cam_col:
        st.markdown("<div style='padding:16px 0 0 16px'>", unsafe_allow_html=True)
        frame_ph  = st.empty()
        status_ph = st.empty()
        live_ph   = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div style='padding:16px 16px 0 8px'>
            <div style='font-size:.65rem;font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;color:#8a96a8;
                        font-family:"JetBrains Mono",monospace;margin-bottom:10px'>
                Controls
            </div>
        """, unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2, gap="small")
        with r1c1: btn_start  = st.button("▶  Start",           key="start")
        with r1c2: btn_reauth = st.button("↺  Reauthenticate",  key="ra")

        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

        r2c1, r2c2, r2c3 = st.columns(3, gap="small")
        with r2c1: btn_atk = st.button("⚡ Attack", key="ta")
        with r2c2: btn_eup = st.button("ε  ＋",    key="eu")
        with r2c3: btn_edn = st.button("ε  −",     key="ed")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Attack status pill
        atk_bg   = "#fef2f2" if st.session_state.attack_on else "#f0fdf4"
        atk_bdr  = "#fecaca" if st.session_state.attack_on else "#bbf7d0"
        atk_col  = "#dc2626" if st.session_state.attack_on else "#059669"
        atk_txt  = "ATTACK ON" if st.session_state.attack_on else "ATTACK OFF"
        st.markdown(f"""
        <div style='background:{atk_bg};border:1.5px solid {atk_bdr};border-radius:9px;
                    padding:10px 14px;display:flex;justify-content:space-between;align-items:center'>
            <span style='font-size:.78rem;font-weight:700;color:{atk_col};
                         font-family:"JetBrains Mono",monospace;letter-spacing:.05em'>
                {atk_txt}
            </span>
            <span style='font-size:.74rem;color:#8a96a8;font-family:"JetBrains Mono",monospace'>
                ε = {st.session_state.epsilon:.3f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:16px;font-size:.58rem;font-weight:700;letter-spacing:2.5px;
                    text-transform:uppercase;color:#8a96a8;
                    font-family:"JetBrains Mono",monospace;margin-bottom:8px'>
            Enrolled Users
        </div>
        """, unsafe_allow_html=True)
        db_ph = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Button handlers ──────────────────────────────────────────
    if btn_start:
        reset_auth()
        st.session_state.auth_active = True

    if btn_reauth:
        st.session_state.decision_locked = False
        st.session_state.locked_result   = ""
        st.session_state.locked_score    = 0.0
        st.session_state.score_history.clear()
        st.session_state.auth_active     = True

    if btn_atk:
        st.session_state.attack_on = not st.session_state.attack_on
        st.session_state.score_history.clear()

    if btn_eup:
        st.session_state.epsilon = min(st.session_state.epsilon + 0.005, 0.1)

    if btn_edn:
        st.session_state.epsilon = max(st.session_state.epsilon - 0.005, 0.001)

    def render_db():
        if not st.session_state.db:
            db_ph.markdown(
                "<div style='font-size:.73rem;color:#8a96a8;font-family:\"Syne\",sans-serif;"
                "padding:8px 0'>No users enrolled yet.</div>",
                unsafe_allow_html=True)
        else:
            rows = "".join(
                f'<div class="db-row"><span class="db-name">{u}</span>'
                f'<span class="db-tpl">{len(t)} tpl</span></div>'
                for u,t in st.session_state.db.items())
            db_ph.markdown(rows, unsafe_allow_html=True)
    render_db()

    # ══════════════════════════════════════════════════════════════
    # MAIN AUTH LOOP
    # ══════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = st.session_state.face_mesh.process(rgb)

        display_msg   = "READY — Press Start to authenticate"
        current_score = 0.0
        live_state    = "IDLE"
        ex=ey=ew=eh=pad=None

        if results.multi_face_landmarks:
            lm  = results.multi_face_landmarks[0]
            p1  = np.array([lm.landmark[468].x*w, lm.landmark[468].y*h])
            p2  = np.array([lm.landmark[473].x*w, lm.landmark[473].y*h])
            pad = int(np.linalg.norm(p1-p2)*0.35)
            pts = np.array([(int(lm.landmark[i].x*w),int(lm.landmark[i].y*h))
                            for i in [33,133,159,145,153,154]])
            ex,ey,ew,eh = cv2.boundingRect(pts)
            crop = frame[max(0,ey-pad):min(h,ey+eh+pad),
                         max(0,ex-pad):min(w,ex+ew+pad)]

            if not st.session_state.auth_active:
                display_msg = "FACE DETECTED — Press ▶ Start"
                live_state  = "IDLE"

            elif st.session_state.decision_locked:
                display_msg   = st.session_state.locked_result
                current_score = st.session_state.locked_score
                live_state    = "LIVE" if "SPOOF" not in display_msg else "SPOOF"

            elif crop.size > 0:
                if not st.session_state.liveness_passed:
                    live = st.session_state.liveness_detector.check(frame, lm)
                    if live is None:
                        display_msg = "VERIFYING LIVENESS..."
                        live_state  = "CHECKING"
                    elif live is False:
                        display_msg = "ACCESS DENIED — SPOOF DETECTED"
                        live_state  = "SPOOF"
                        st.session_state.decision_locked = True
                        st.session_state.locked_result   = display_msg
                        st.session_state.locked_score    = 0.0
                    else:
                        st.session_state.liveness_passed = True
                        live_state = "LIVE"

                if st.session_state.liveness_passed and not st.session_state.decision_locked:
                    live_state = "LIVE"
                    enhanced   = enhance_crop(crop)
                    pil_img    = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                    tensor     = st.session_state.transform(pil_img).unsqueeze(0).to(st.session_state.device)

                    if st.session_state.attack_on:
                        tensor = fgsm_attack(st.session_state.model, tensor,
                                             st.session_state.epsilon)
                        sig = get_sig_from_tensor(tensor)
                    else:
                        sig = get_signature(crop, st.session_state.model,
                                            st.session_state.transform)
                    if st.session_state.device.type == "cuda":
                        torch.cuda.empty_cache()

                    if st.session_state.db:
                        success, user, current_score = authenticate(
                            sig, st.session_state.db, st.session_state.score_history)
                        if success:
                            display_msg = f"ACCESS GRANTED: {user}"
                            st.session_state.decision_locked = True
                            st.session_state.locked_result   = display_msg
                            st.session_state.locked_score    = current_score
                        else:
                            display_msg = "ACCESS DENIED"
                            st.session_state.decision_locked = True
                            st.session_state.locked_result   = display_msg
                            st.session_state.locked_score    = current_score
                    else:
                        display_msg = "NO DATABASE — Please enrol first"

        else:
            reset_auth()
            display_msg = "SCANNING..."
            live_state  = "IDLE"

        # ── Draw ROI brackets ─────────────────────────────────────
        if ex is not None:
            if   "GRANTED"  in display_msg: bc=(5,150,105)
            elif "SPOOF"    in display_msg: bc=(217,119,6)
            elif "DENIED"   in display_msg: bc=(220,38,38)
            elif "CHECKING" in live_state:  bc=(37,99,235)
            elif not st.session_state.auth_active: bc=(138,150,168)
            else: bc=(79,70,229)
            bx1,by1,bx2,by2 = ex-pad,ey-pad,ex+ew+pad,ey+eh+pad
            seg=18
            for pts2 in [
                [(bx1,by1+seg),(bx1,by1),(bx1+seg,by1)],
                [(bx2-seg,by1),(bx2,by1),(bx2,by1+seg)],
                [(bx1,by2-seg),(bx1,by2),(bx1+seg,by2)],
                [(bx2-seg,by2),(bx2,by2),(bx2,by2-seg)],
            ]:
                cv2.polylines(frame,[np.array(pts2)],False,bc,2,cv2.LINE_AA)

        # ── Status bar ────────────────────────────────────────────
        if   "GRANTED"  in display_msg: sb_cls = "sb-granted"
        elif "SPOOF"    in display_msg: sb_cls = "sb-spoof"
        elif "DENIED"   in display_msg: sb_cls = "sb-denied"
        elif "VERIFY"   in display_msg: sb_cls = "sb-scan"
        elif "LIVENESS" in display_msg: sb_cls = "sb-scan"
        else:                           sb_cls = "sb-idle"

        sc_col = ("#059669" if current_score>=MATCH_THRESHOLD
                  else "#d97706" if current_score>=0.55 else "#dc2626")

        status_ph.markdown(f"""
        <div style='margin:6px 0 0'>
        <div class='sb {sb_cls}'>
            <span style='font-size:1.3rem;font-weight:800;letter-spacing:0.01em'>{display_msg}</span>
            <div class='sb-right'>
                <span style='color:#8a96a8;font-size:.85rem;font-weight:600'>SIM</span>
                <span class='sb-score' style='color:{sc_col}'>{current_score:.4f}</span>
                <span style='color:#d0d6e0'>|</span>
                <span style='color:{"#dc2626" if st.session_state.attack_on else "#059669"};
                             font-size:.95rem;font-weight:800'>
                    {"ATK ON" if st.session_state.attack_on else "ATK OFF"}
                </span>
                <span style='color:#8a96a8;font-size:.85rem;font-weight:600'>ε={st.session_state.epsilon:.3f}</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Liveness row ──────────────────────────────────────────
        ld_map = {
            "LIVE":     ("#059669", "#ecfdf5", "#a7f3d0", "LIVE"),
            "CHECKING": ("#2563eb", "#eff6ff", "#bfdbfe", "CHECKING"),
            "SPOOF":    ("#d97706", "#fffbeb", "#fde68a", "SPOOF"),
            "IDLE":     ("#8a96a8", "#f8f9fb", "#e2e6ed", "IDLE"),
        }.get(live_state, ("#8a96a8", "#f8f9fb", "#e2e6ed", live_state))

        locked_tag = '<span style="margin-left:auto;color:#8a96a8;font-size:.75rem">LOCKED — press Start / Reauthenticate</span>' if st.session_state.decision_locked else ''
        live_ph.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin:5px 0 0;
                    font-family:"JetBrains Mono",monospace;font-size:.72rem'>
            <div style='width:7px;height:7px;border-radius:50%;
                        background:{ld_map[0]};box-shadow:0 0 0 3px {ld_map[2]}'></div>
            <span style='color:{ld_map[0]};font-weight:600'>LIVENESS: {ld_map[3]}</span>
            {locked_tag}
        </div>
        """, unsafe_allow_html=True)

        disp = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        frame_ph.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB),
                       use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — ENROLLMENT
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "ENROLL":

    st.markdown("""
    <div style='padding:22px 28px 12px;border-bottom:1.5px solid #e2e6ed;
                background:#fff;margin-bottom:0'>
        <div style='display:flex;align-items:baseline;gap:14px'>
            <span style='font-size:1.45rem;font-weight:800;color:#0f1117;
                         font-family:"Syne",sans-serif;letter-spacing:-0.02em'>Enrollment</span>
            <span style='font-size:.58rem;color:#8a96a8;font-family:"JetBrains Mono",monospace;
                         letter-spacing:2.5px;text-transform:uppercase'>
                Biometric Template Registration
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_cam = st.columns([1, 2], gap="large")

    with col_form:
        st.markdown("<div style='padding:16px 8px 0 20px'>", unsafe_allow_html=True)

        st.markdown("<div style='color:#4a5568;font-size:.75rem;font-weight:600;"
                    "font-family:\"Syne\",sans-serif;margin-bottom:5px'>User Name</div>",
                    unsafe_allow_html=True)
        e_name  = st.text_input("", placeholder="Enter full name",
                                label_visibility="collapsed", key="ename")
        e_specs = st.checkbox("User wears glasses", key="especs")

        st.markdown("""
        <div style='background:#eef0ff;border:1.5px solid #c7d2fe;border-radius:10px;
                    padding:13px 15px;font-size:.76rem;color:#4338ca;
                    line-height:2;margin:12px 0;font-family:"Syne",sans-serif'>
            <strong>Phase 1</strong> — 30 captures, no glasses<br>
            <strong style='color:#059669'>Phase 2</strong> — 30 captures with glasses
            <em style='color:#8a96a8'> (if selected)</em><br>
            <span style='color:#8a96a8;font-size:.68rem;font-family:"JetBrains Mono",monospace'>
                Each capture → 6 lighting variants
            </span>
        </div>
        """, unsafe_allow_html=True)

        btn_start_enr = st.button("▶  Start Enrollment", key="enr")

        st.markdown("""
        <div style='margin-top:18px;font-size:.58rem;font-weight:700;letter-spacing:2.5px;
                    text-transform:uppercase;color:#8a96a8;
                    font-family:"JetBrains Mono",monospace;margin-bottom:8px'>
            Enrolled Users
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.db:
            for u,t in st.session_state.db.items():
                st.markdown(
                    f'<div class="db-row"><span class="db-name">{u}</span>'
                    f'<span class="db-tpl">{len(t)} templates</span></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:.73rem;color:#8a96a8;"
                        "font-family:\"Syne\",sans-serif;padding:6px 0'>No users enrolled yet.</div>",
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_cam:
        st.markdown("<div style='padding:16px 20px 0 0'>", unsafe_allow_html=True)
        cam_e_lbl = st.empty()
        frame_e   = st.empty()
        phase_e   = st.empty()
        prog_e    = st.empty()
        cam_e_lbl.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:.58rem;"
            "font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#8a96a8;"
            "margin-bottom:6px'>Enrollment Camera</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if btn_start_enr and e_name.strip():
        uname  = e_name.strip()
        TARGET = 60
        st.session_state.db[uname] = []

        def run_phase(label, badge_cls, has_specs_flag, base_count):
            captured = 0
            while captured < TARGET:
                ret, frame = cap.read()
                if not ret: break
                h,w,_ = frame.shape
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res    = st.session_state.face_mesh.process(rgb)

                if res.multi_face_landmarks:
                    lm  = res.multi_face_landmarks[0]
                    p1  = np.array([lm.landmark[468].x*w,lm.landmark[468].y*h])
                    p2  = np.array([lm.landmark[473].x*w,lm.landmark[473].y*h])
                    pad = int(np.linalg.norm(p1-p2)*0.35)
                    pts = np.array([(int(lm.landmark[i].x*w),int(lm.landmark[i].y*h))
                                    for i in [33,133,159,145,153,154]])
                    ex,ey,ew,eh = cv2.boundingRect(pts)
                    crop = frame[max(0,ey-pad):min(h,ey+eh+pad),
                                 max(0,ex-pad):min(w,ex+ew+pad)]
                    if crop.size > 0:
                        _, captured, _ = handle_enrollment(
                            crop, uname, st.session_state.db, captured,
                            has_specs_flag, TARGET,
                            lambda c: get_signature(c, st.session_state.model,
                                                    st.session_state.transform))
                    bx1,by1,bx2,by2 = ex-pad,ey-pad,ex+ew+pad,ey+eh+pad
                    for pts2 in [
                        [(bx1,by1+14),(bx1,by1),(bx1+14,by1)],
                        [(bx2-14,by1),(bx2,by1),(bx2,by1+14)],
                        [(bx1,by2-14),(bx1,by2),(bx1+14,by2)],
                        [(bx2-14,by2),(bx2,by2),(bx2,by2-14)],
                    ]:
                        cv2.polylines(frame,[np.array(pts2)],False,(79,70,229),2,cv2.LINE_AA)

                pct = int(captured/TARGET*100)
                phase_e.markdown(
                    f'<span class="ep-badge {badge_cls}">'
                    f'{label} &nbsp; {captured} / {TARGET}</span>',
                    unsafe_allow_html=True)
                prog_e.markdown(f"""
                <div class='ep-track'><div class='ep-fill' style='width:{pct}%'></div></div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:.62rem;
                            color:#8a96a8;margin-top:4px'>
                    {(base_count+captured)*6} templates stored
                </div>""", unsafe_allow_html=True)
                disp_e = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                frame_e.image(cv2.cvtColor(disp_e, cv2.COLOR_BGR2RGB),
                              use_container_width=True)
                time.sleep(0.05)

        run_phase("PHASE 1 — MOVE HEAD SLOWLY", "ep-p1", False, 0)

        if e_specs:
            for i in range(5, 0, -1):
                phase_e.markdown(
                    f'<span class="ep-badge ep-paus">PUT ON GLASSES — Phase 2 in {i}s</span>',
                    unsafe_allow_html=True)
                time.sleep(1)
            run_phase("PHASE 2 — WEAR SPECS & MOVE", "ep-p2", True, TARGET)

        save_database(st.session_state.db)
        total = len(st.session_state.db.get(uname, []))
        phase_e.markdown(
            f'<span class="ep-badge ep-done">✓ Done — {uname} ({total} templates)</span>',
            unsafe_allow_html=True)
        prog_e.progress(100, text="Saved to database")

# cap stays open — never release mid-session