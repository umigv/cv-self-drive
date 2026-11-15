import cv2
import numpy as np
from hsv import hsv
from functional_test_parent import FunctionalTest

"""
Test FV.1 Unobstructed STATIC pedestrian detection (2026 rules, page 36)

Goal: Ego vehicle stops if a pedestrian (mannequin) is detected within the
boundaries of the current lane.

Script:
    1. Vehicle takes off from full stop at Barrel 1
    2. Vehicle maintains target speed (4 - 5 mph)
    3. Vehicle reaches full stop 5 ft - 7 ft from the mannequin
    4. End test run

Fail: stops further than 7 ft from the mannequin, or hits the mannequin
Penalty: stops closer than 5 ft from the mannequin (10 points)

Adapted from ObstructedPedestrianDetection. Since the pedestrian is STATIC and
the run ends once the vehicle stops, there is no "resume lane keeping" state.
"""

class UnobstructedPedestrianStop(FunctionalTest):
    def __init__(self, pedestrian_model=None, stop_size_threshold=0.12, confidence_threshold=0.7):
        super().__init__()
        self.image = None
        self.height = 0
        self.width = 0
        self.hsv_obj = None
        self.pedestrian_model = pedestrian_model
        self.state = 1  # Start in lane keeping state
        self.white_mask = None
        self.yellow_mask = None
        self.waypoint = (0, 0)

        # Pedestrian bounding box width / frame width at which we are 5 - 7 ft away.
        # Tune this on the course so the vehicle stops inside the 5 - 7 ft window.
        self.stop_size_threshold = stop_size_threshold
        self.confidence_threshold = confidence_threshold

    def detect_static_pedestrian(self):
        """Returns (in_stop_range, py2, px1) for the closest pedestrian in the current lane."""
        if self.pedestrian_model is None:
            return False, 0, 0

        results = self.pedestrian_model(self.image)
        py2 = 0
        px1 = 0

        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.tolist()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                # class 0 is person (built in) and adjust confidence as needed
                if class_id != 0 or confidence < self.confidence_threshold:
                    continue

                px1, py1, px2, py2 = map(int, box)
                if not self.in_current_lane(px1, px2, py2):
                    continue

                y_range = int(self.width / 100)
                size_person = (px2 - px1) / self.width
                cv2.rectangle(self.image, (px1, py1), (px2, py2), (0, 255, 0), 2)
                if size_person > self.stop_size_threshold:
                    # print("person within range")
                    return True, (py2 + y_range), px1
                else:
                    return False, (py2 + y_range), px1

        return False, py2, px1

    def in_current_lane(self, px1, px2, py2):
        """Checks if the pedestrian's feet are between the lane lines of the current lane."""
        center_x = (px1 + px2) // 2
        left_x, right_x = self.find_lane_bounds(min(py2, self.height - 1))
        return left_x <= center_x <= right_x

    def find_lane_bounds(self, row):
        """Finds the closest lane line pixels left and right of center on the given row.
        Falls back to the middle half of the frame if a line is not found."""
        mid = self.width // 2
        left_x = self.width // 4
        right_x = self.width - self.width // 4

        if self.final_mask is None:
            return left_x, right_x

        line_pixels = np.where(self.final_mask[row, :] > 0)[0]
        left_pixels = line_pixels[line_pixels < mid]
        right_pixels = line_pixels[line_pixels >= mid]

        if left_pixels.size > 0:
            left_x = int(np.max(left_pixels))
        if right_pixels.size > 0:
            right_x = int(np.min(right_pixels))

        return left_x, right_x

    def state_machine(self):
        # State 1: lane keeping + pedestrian search
        # State 2: pedestrian within 5 - 7 ft -> full stop (end of test run)

        if self.state == 1:
            self.waypoint = (self.width // 2, 40)
            pedestrian_detected, py2, px1 = self.detect_static_pedestrian()
            if pedestrian_detected:
                print("PEDESTRIAN IN LANE - STOPPING")
                self.state = 2

        if self.state == 2:
            # Pedestrian is static and the run ends at the stop, so stay stopped
            self.waypoint = (self.width // 2, self.height)

    def update_mask(self):
        mask, dict = self.hsv_obj.get_mask(self.image)
        self.white_mask = dict.get("white")
        self.yellow_mask = dict.get("yellow")
        self.final_mask = mask

    def main(self):
        from ultralytics import YOLO

        if self.pedestrian_model is None:
            self.pedestrian_model = YOLO("./data/yolov8n.pt")

        cap = cv2.VideoCapture("placeholder.mp4")
        self.hsv_obj = hsv("placeholder.mp4")

        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape

                self.update_mask()
                self.state_machine()

                cv2.circle(self.final_mask, self.waypoint, 5, 255, -1)
                cv2.imshow("Unobstructed Pedestrian Stop", self.final_mask)
                cv2.imshow("Pedestrian Detection", self.image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_frame(self, hsv_indentifier, frame):
        if self.hsv_obj is None:
            self.hsv_obj = hsv(hsv_indentifier)

        self.image = frame
        self.height, self.width, _ = self.image.shape

        self.update_mask()
        self.state_machine()

        cv2.imshow("Unobstructed Pedestrian Stop", self.final_mask)

        return self.final_mask, self.waypoint


if __name__ == "__main__":
    test = UnobstructedPedestrianStop()
    test.main()
