from camera_feed import start_camera_feed
from deepfake_lips import start_lip_sync_camera_feed

def main():
    print("Starting the camera feed with lip sync using DeepFaceLive...")
    start_lip_sync_camera_feed('right.MOV')

if __name__ == "__main__":
    main()
