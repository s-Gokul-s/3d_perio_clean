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
# INITIALIZATION
# ==============================
model = load_model()
transform = get_transform()
db = load_database()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

cv2.namedWindow("Biometric Access Control",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Biometric Access Control",640,480)

liveness_detector = LivenessDetector()

mode="AUTH"
enroll_step="IDLE"
enroll_name=""
has_specs=False
samples_captured=0
score_history=[]

pause_start=None


# ==============================
# DECISION LOCK
# ==============================
decision_locked=False
locked_result=""
locked_score=0.0


# ==============================
# AUTO RESET WHEN FACE LEAVES
# ==============================
NO_FACE_RESET_FRAMES=30
no_face_counter=0


# ==============================
# ADVERSARIAL ATTACK
# ==============================
attack_on=False
epsilon=0.005


# ==============================
# FGSM ATTACK
# ==============================
def fgsm_attack(model,tensor,epsilon):

    tensor=tensor.clone().detach().to(device)
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
        out=model.vit(tensor)

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
    results=face_mesh.process(rgb)

    status_color=(255,255,0)
    display_msg=f"SYSTEM: {mode}"
    info_msg="PRESS [E] TO ENROLL"
    current_score=0.0

    ex=ey=ew=eh=pad=None

    if results.multi_face_landmarks:

        no_face_counter=0

        lm=results.multi_face_landmarks[0]

        p1=np.array([lm.landmark[468].x*w,lm.landmark[468].y*h])
        p2=np.array([lm.landmark[473].x*w,lm.landmark[473].y*h])

        dist=np.linalg.norm(p1-p2)
        pad=int(dist*0.35)

        eye_idx=[33,133,159,145,153,154]

        pts=np.array([
            (int(lm.landmark[i].x*w),
             int(lm.landmark[i].y*h))
            for i in eye_idx
        ])

        ex,ey,ew,eh=cv2.boundingRect(pts)

        crop=frame[
            max(0,ey-pad):min(h,ey+eh+pad),
            max(0,ex-pad):min(w,ex+ew+pad)
        ]

    else:

        crop=None
        no_face_counter+=1

        if no_face_counter>NO_FACE_RESET_FRAMES:

            decision_locked=False
            score_history.clear()
            liveness_detector.reset()

            display_msg="SYSTEM RESET"
            status_color=(0,165,255)


    # ==============================
    # DECISION LOCK
    # ==============================
    if decision_locked:

        display_msg=locked_result
        current_score=locked_score

        if "GRANTED" in locked_result:
            status_color=(0,255,0)
        else:
            status_color=(0,0,255)

        info_msg="PRESS R TO RE-AUTHENTICATE"

    else:

        if crop is None or crop.size==0:

            display_msg="NO FACE DETECTED"
            status_color=(0,0,255)

        else:

            roi_display=enhance_crop(crop)
            cv2.imshow("Periocular ROI",roi_display)

            if mode=="AUTH":

                live=liveness_detector.check(frame,lm)

                if live is None:

                    display_msg="VERIFYING LIVENESS..."
                    status_color=(0,255,255)

                elif live is False:

                    display_msg="ACCESS DENIED - SPOOF DETECTED"
                    status_color=(0,0,255)

                    decision_locked=True
                    locked_result=display_msg
                    locked_score=0

                else:

                    enhanced=enhance_crop(crop)

                    pil_img=Image.fromarray(
                        cv2.cvtColor(enhanced,cv2.COLOR_BGR2RGB)
                    )

                    tensor=transform(pil_img).unsqueeze(0).to(device)

                    if attack_on:
                        tensor=fgsm_attack(model,tensor,epsilon)
                        sig=get_signature_from_tensor(tensor)
                    else:
                        sig=get_signature(crop,model,transform)

                    if db:

                        success,user,current_score=authenticate(
                            sig,db,score_history
                        )

                        if success:

                            display_msg=f"ACCESS GRANTED: {user}"
                            status_color=(0,255,0)

                            decision_locked=True
                            locked_result=display_msg
                            locked_score=current_score

                        else:

                            display_msg="ACCESS DENIED"
                            status_color=(0,0,255)

                            decision_locked=True
                            locked_result=display_msg
                            locked_score=current_score


            elif mode=="ENROLL":

                if enroll_step=="WAITING_FOR_SPACE":

                    info_msg="LOOK AT CAMERA & PRESS SPACE"
                    status_color=(0,255,255)

                elif enroll_step=="PHASE1":

                    finished,samples_captured,info_msg=handle_enrollment(
                        crop,
                        enroll_name,
                        db,
                        samples_captured,
                        False,
                        60,
                        lambda c:get_signature(c,model,transform)
                    )

                    if finished:

                        if has_specs:
                            enroll_step="PAUSE"
                            pause_start=time.time()
                        else:
                            save_database(db)
                            mode="AUTH"
                            enroll_step="IDLE"

                elif enroll_step=="PAUSE":

                    display_msg="PLEASE WEAR GLASSES"
                    info_msg="Enrollment resumes in 5 seconds"

                    if time.time()-pause_start>5:

                        samples_captured=0
                        enroll_step="PHASE2"

                elif enroll_step=="PHASE2":

                    finished,samples_captured,info_msg=handle_enrollment(
                        crop,
                        enroll_name,
                        db,
                        samples_captured,
                        True,
                        60,
                        lambda c:get_signature(c,model,transform)
                    )

                    if finished:

                        save_database(db)
                        mode="AUTH"
                        enroll_step="IDLE"


    # ==============================
    # DRAW ROI BOX
    # ==============================
    if ex is not None:

        box_color=(0,0,255) if attack_on else status_color

        cv2.rectangle(
            frame,
            (ex-pad,ey-pad),
            (ex+ew+pad,ey+eh+pad),
            box_color,
            2
        )


    # ==============================
    # UI
    # ==============================
    cv2.rectangle(frame,(0,0),(w,70),(20,20,20),-1)

    score_color=(0,255,0) if current_score> MATCH_THRESHOLD else (0,0,255)

    cv2.putText(frame,
                f"Similarity: {current_score:.4f}",
                (w-320,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                score_color,
                2)

    cv2.putText(frame,
                display_msg,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2)

    cv2.putText(frame,
                info_msg,
                (20,h-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,200,200),
                1)


    # ==============================
    # ATTACK STATUS UI (RESTORED)
    # ==============================
    if attack_on:

        cv2.rectangle(frame,(0,70),(w,110),(0,0,180),-1)

        cv2.putText(frame,
                    f"FGSM ATTACK ACTIVE   epsilon={epsilon:.4f}   Press A to disable",
                    (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

    else:

        cv2.rectangle(frame,(0,70),(w,110),(0,120,0),-1)

        cv2.putText(frame,
                    f"Attack OFF   epsilon={epsilon:.4f}   Press A to enable attack",
                    (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)


    cv2.imshow("Biometric Access Control",frame)


    # ==============================
    # KEYBOARD CONTROLS
    # ==============================
    key=cv2.waitKey(1)&0xFF

    if key==ord('q'):
        break

    elif key==ord('r'):
        decision_locked=False
        score_history.clear()

    elif key==ord('a'):
        attack_on=not attack_on
        score_history.clear()

    elif key==ord('+') or key==ord('='):
        epsilon=min(epsilon+0.005,0.1)

    elif key==ord('-'):
        epsilon=max(epsilon-0.005,0.001)

    elif key==ord('e') and mode=="AUTH":

        enroll_name=input("Enter User Name: ")
        specs_input=input("Does this user wear glasses? (y/n): ").lower()

        has_specs=specs_input=='y'

        db[enroll_name]=[]
        samples_captured=0

        mode="ENROLL"
        enroll_step="WAITING_FOR_SPACE"

    elif key==32 and enroll_step=="WAITING_FOR_SPACE":
        enroll_step="PHASE1"


cap.release()
cv2.destroyAllWindows()