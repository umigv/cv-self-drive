import cv2 as cv
import numpy as np
import time

from functional_test_parent import FunctionalTest

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

'''
1. get frame
2. detect the objects
'''

frame_number = 1
image_path = f"data/frames/frame_{frame_number}.png"

class ObstacleLaneChange(FunctionalTest):
    def __init__(self): 
        super().__init__
        obj_detection_model = None

    def useYoloModel(self, image):
        t0 = time.time() #begin of inference
        results = self.obj_detection_model.predict(source=image, project="data/results", name=f"result_{frame_number}.png")
        t = time.time() #end of inference

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.save(filename=f"data/results/result_{frame_number}.png")  # save thing

    def run_frame():
        pass
    

def main():
    '''
    to get a waypoint, we first
    1. get the images
    2. run yolo model on it
    3. the closest image in the lane we are in is the problem, go away from it when we get too close
    4. get waypoint to go away from it
    '''
    obstacles = ObstacleLaneChange()
    current_image = cv.imread(image_path)

    if current_image is None:
        print("error reading image")
        exit()

    weights = hf_hub_download("Ultralytics/YOLOv8", "yolov8l.pt")
    yolo_model = YOLO(weights)

    obstacles.obj_detection_model = yolo_model
    obstacles.useYoloModel(current_image)

    for i in range(643):
        global frame_number 
        frame_number = frame_number + 1
        current_image = cv.imread(f"data/frames/frame_{frame_number}.png")
        obstacles.useYoloModel(current_image)

if __name__ == "__main__":
    main()