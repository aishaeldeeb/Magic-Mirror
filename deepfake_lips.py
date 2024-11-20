from pathlib import Path
import cv2
import numpy as np
import subprocess
from DeepFaceLive.apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp
import dlib
from DeepFaceLive.modelhub.onnx import InsightFaceSwap

from DeepFaceLive.apps.DeepFaceLive.backend.FaceAligner import FaceAligner
from DeepFaceLive.apps.DeepFaceLive.backend.FaceAnimator import FaceAnimator
from DeepFaceLive.apps.DeepFaceLive.backend.FaceSwapInsight import FaceSwapInsight
from DeepFaceLive.apps.DeepFaceLive.backend.BackendBase import BackendConnectionData, BackendSignal, BackendWeakHeap, BackendConnection, BackendDB

# Load pre-trained facial landmark model
predictor_path = "./facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define paths and shared backend objects
userdata_path = Path("/home/aisha/deepfake_project/CPEN541/DeepFaceLive")
faces_path = userdata_path / "faces"
animatables_path = userdata_path / "animatables"
weak_heap = BackendWeakHeap(size_mb=512)
reemit_frame_signal = BackendSignal()
bc_in = BackendConnection(multi_producer=True)
bc_out = BackendConnection(multi_producer=False)
backend_db = BackendDB()

# Initialize FaceAligner
aligner = FaceAligner(
    weak_heap=weak_heap, 
    reemit_frame_signal=reemit_frame_signal, 
    bc_in=bc_in, 
    bc_out=bc_out,
    backend_db=backend_db
)

# Initialize FaceAnimator
animator = FaceAnimator(
    weak_heap=weak_heap, 
    reemit_frame_signal=reemit_frame_signal, 
    bc_in=bc_in, 
    bc_out=bc_out, 
    animatables_path=animatables_path,
    backend_db=backend_db,
    id=1  # Example ID; adjust as needed
)

# Initialize FaceSwapInsight
face_swap = FaceSwapInsight(
    weak_heap=weak_heap, 
    reemit_frame_signal=reemit_frame_signal, 
    bc_in=bc_in, 
    bc_out=bc_out, 
    faces_path=faces_path,
    backend_db=backend_db,
    id=2  # Example ID; adjust as needed
)

# Initialize the device for InsightFaceSwap
print(f"Availables devices: {InsightFaceSwap.get_available_devices()}")
device_info = InsightFaceSwap.get_available_devices()[0]
print(f"Selected device: {device_info}")

face_swap.swap_model = InsightFaceSwap(device_info=device_info)

def start_deepface_live():
    subprocess.Popen(["./deepface_venv/Scripts/python.exe", "main.py", "run", "DeepFaceLive", "--userdata-dir", str(userdata_path)])

def load_lip_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_lip_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None
    
    for face in faces:
        landmarks = predictor(gray, face)
        points = []
        for n in range(48, 68):  # Points around the lips
            points.append((landmarks.part(n).x, landmarks.part(n).y))
        return points

def apply_lip_sync(frame, lip_frames, landmarks):
    # Ensure the frame size is manageable
    target_size = (640, 480)  # Example resolution, adjust as needed
    frame_resized = cv2.resize(frame, target_size)
    lip_frame = lip_frames[0]
    lip_frame_resized = cv2.resize(lip_frame, target_size)


    if landmarks is None:
        return frame

    # Resize the lip frame to match the dimensions of the original frame
    lip_frame_resized = cv2.resize(lip_frame, (frame.shape[1], frame.shape[0]))

    # Use FaceAnimator for animated adjustments based on motion
    animator.driving_ref_motion = landmarks  # Integrate landmark motion data

    alpha = 0.4
    blended_frame = cv2.addWeighted(frame, 1 - alpha, lip_frame_resized, alpha, 0)

    # Apply a slight Gaussian blur to reduce hard edges
    blended_frame = cv2.GaussianBlur(blended_frame, (5, 5), 0)
    return blended_frame

def start_lip_sync_camera_feed(lip_video_path):
    lip_frames = load_lip_video(lip_video_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    app = DeepFaceLiveApp(userdata_path=userdata_path)
    backend_data = BackendConnectionData(uid=1)  # Example UID; adjust if necessary

    # Generate a face vector from the reference frame (first frame of lip sync video)
    reference_frame = lip_frames[0]
    face_vector = face_swap.swap_model.get_face_vector(reference_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        landmarks = get_lip_landmarks(frame)
        synced_frame = apply_lip_sync(frame, lip_frames, landmarks)
        
        # Generate the face-swapped output using the face vector
        output_frame = face_swap.swap_model.generate(synced_frame, face_vector)
        
        # Emit the updated frame
        # reemit_frame_signal.emit(output_frame)  # Emit the frame directly
        
        cv2.imshow('Lip Sync Camera Feed', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_deepface_live()
    start_lip_sync_camera_feed('right.mov')