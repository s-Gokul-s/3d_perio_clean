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


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI Periocular Biometric System",
    layout="wide"
)

st.markdown(
"""
<style>
.main {background-color:#f3f6fb;}
.status-good {color:#1bb55c;font-size:22px;font-weight:600;}
.status-bad {color:#e53935;font-size:22px;font-weight:600;}
.status-warn {color:#ff9800;font-size:22px;font-weight:600;}
.title {font-size:32px;font-weight:700;margin-bottom:10px;}
</style>
""",
unsafe_allow_html=True
)

st.markdown("<div class='title'>AI Periocular Biometric Authentication System</div>", unsafe_allow_html=True)


# ==============================
# SESSION STATE INIT
# ==============================
if "initialized" not in st.session_state:

    st.session_state.model = load_model()
    st.session_state.transform = get_transform()
    st.session_state.db = load_database()

    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state.model.eval()

    st.session_state.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    st.session_state.liveness_detector = LivenessDetector()

    st.session_state.mode="AUTH"
    st.session_state.enroll_step="IDLE"
    st.session_state.enroll_name=""
    st.session_state.has_specs=False
    st.session_state.samples_captured=0
    st.session_state.score_history=[]

    st.session_state.decision_locked=False
    st.session_state.locked_result=""
    st.session_state.locked_score=0.0

    st.session_state.attack_on=False
    st.session_state.epsilon=0.005

    st.session_state.no_face_counter=0
    st.session_state.initialized=True


# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("System Controls")

if st.sidebar.button("Reauthenticate"):
    st.session_state.decision_locked=False
    st.session_state.score_history.clear()

if st.sidebar.button("Toggle Attack"):
    st.session_state.attack_on = not st.session_state.attack_on
    st.session_state.score_history.clear()

if st.sidebar.button("Increase Epsilon"):
    st.session_state.epsilon = min(st.session_state.epsilon + 0.005, 0.1)

if st.sidebar.button("Decrease Epsilon"):
    st.session_state.epsilon = max(st.session_state.epsilon - 0.005, 0.001)

st.sidebar.markdown(f"**Attack Status:** {'ON' if st.session_state.attack_on else 'OFF'}")
st.sidebar.markdown(f"**Epsilon:** {st.session_state.epsilon:.4f}")

if st.sidebar.button("Start Enrollment"):

    name = st.sidebar.text_input("User Name")

    if name != "":
        st.session_state.enroll_name=name
        st.session_state.db[name]=[]
        st.session_state.samples_captured=0
        st.session_state.mode="ENROLL"
        st.session_state.enroll_step="PHASE1"


# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()
status_box = st.empty()


# ==============================
# FGSM ATTACK
# ==============================
def fgsm_attack(model,tensor,epsilon):

    tensor=tensor.clone().detach().to(st.session_state.device)
    tensor.requires_grad_(True)

    logits=model(tensor).logits
    label=torch.argmax(logits,dim=1)

    loss=torch.nn.CrossEntropyLoss()(logits,label)

    model.zero_grad()
    loss.backward()

    adv=tensor+epsilon*tensor.grad.sign()

    return torch.clamp(adv,-1,1).detach()


def get_signature_from_tensor(tensor):

    with torch.no_grad():
        out=st.session_state.model.vit(tensor)

    return out.last_hidden_state[0,0].cpu().numpy()


# ==============================
# MAIN LOOP
# ==============================
while True:

    ret,frame=cap.read()
    if not ret:
        break

    h,w,_=frame.shape

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=st.session_state.face_mesh.process(rgb)

    display_msg="System Running"
    status_color="status-warn"
    current_score=0

    ex=ey=ew=eh=pad=None

    if results.multi_face_landmarks:

        st.session_state.no_face_counter=0
        lm=results.multi_face_landmarks[0]

        p1=np.array([lm.landmark[468].x*w,lm.landmark[468].y*h])
        p2=np.array([lm.landmark[473].x*w,lm.landmark[473].y*h])

        dist=np.linalg.norm(p1-p2)
        pad=int(dist*0.35)

        eye_idx=[33,133,159,145,153,154]

        pts=np.array([(int(lm.landmark[i].x*w),int(lm.landmark[i].y*h)) for i in eye_idx])
        ex,ey,ew,eh=cv2.boundingRect(pts)

        crop=frame[max(0,ey-pad):min(h,ey+eh+pad),max(0,ex-pad):min(w,ex+ew+pad)]

    else:

        crop=None
        st.session_state.no_face_counter+=1

        if st.session_state.no_face_counter>30:

            st.session_state.decision_locked=False
            st.session_state.score_history.clear()
            st.session_state.liveness_detector.reset()

            display_msg="System Reset"


    if st.session_state.decision_locked:

        display_msg=st.session_state.locked_result
        current_score=st.session_state.locked_score

        if "GRANTED" in display_msg:
            status_color="status-good"
        else:
            status_color="status-bad"

    else:

        if crop is not None:

            if st.session_state.mode=="AUTH":

                live=st.session_state.liveness_detector.check(frame,lm)

                if live is None:

                    display_msg="Verifying Liveness..."

                elif live is False:

                    display_msg="Spoof Detected"
                    status_color="status-bad"

                    st.session_state.decision_locked=True
                    st.session_state.locked_result=display_msg

                else:

                    enhanced=enhance_crop(crop)

                    pil_img=Image.fromarray(cv2.cvtColor(enhanced,cv2.COLOR_BGR2RGB))

                    tensor=st.session_state.transform(pil_img).unsqueeze(0).to(st.session_state.device)

                    if st.session_state.attack_on:
                        tensor=fgsm_attack(st.session_state.model,tensor,st.session_state.epsilon)
                        sig=get_signature_from_tensor(tensor)
                    else:
                        sig=get_signature(crop,st.session_state.model,st.session_state.transform)

                    success,user,current_score=authenticate(sig,st.session_state.db,st.session_state.score_history)

                    if success:

                        display_msg=f"Access Granted: {user}"
                        status_color="status-good"

                        st.session_state.decision_locked=True
                        st.session_state.locked_result=display_msg
                        st.session_state.locked_score=current_score

                    else:

                        display_msg="Access Denied"
                        status_color="status-bad"

                        st.session_state.decision_locked=True
                        st.session_state.locked_result=display_msg
                        st.session_state.locked_score=current_score


    if ex is not None:

        color=(0,0,255) if st.session_state.attack_on else (0,255,0)

        cv2.rectangle(frame,(ex-pad,ey-pad),(ex+ew+pad,ey+eh+pad),color,2)

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame,use_container_width=True)

    status_box.markdown(
        f"<div class='{status_color}'>Status: {display_msg}<br>Similarity: {current_score:.4f}</div>",
        unsafe_allow_html=True
    )