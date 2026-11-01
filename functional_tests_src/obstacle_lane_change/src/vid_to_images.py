import cv2 as cv

from pathlib import Path

'''
usage:
- make data folder put video in it
- change video path to correct one
'''

def main():
    #i want to take video and then make images from that video
    video_path = "data/obj_detection_lanes.mov"
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("bad file path")
        exit()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count = frame_count + 1

        if ret:
            frame_path = Path(f"data/frames/frame_{frame_count}.png")
            cv.imwrite(frame_path, frame) #write frame
        else:
            print("no frame")
            break #ok we are done
    cap.release()


if __name__ == "__main__":
    main()