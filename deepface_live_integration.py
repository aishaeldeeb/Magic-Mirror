import cv2
import subprocess
from camera_feed import start_camera_feed 

def start_deepface_live():
    subprocess.run(["DeepFaceLive.bat"])

def main():
    print("Starting the camera feed with DeepFaceLive...")
    start_camera_feed()
    start_deepface_live()

if __name__ == "__main__":
    main()
