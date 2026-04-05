from functional_test_parent import FunctionalTest

class ObstructedPedestrianDetection(FunctionalTest):
    def __init__(self):
        super().__init__()
        self.image = None
        self.height = 0
        self.width = 0
        self.hsv_obj = None
        self.pedestrian_model = None
        self.state = 1  # Start in lane keeping state

    def detect_pedestrian(self):
        pass

    def find_lane(self):
        pass
    
    def calculate_waypoint(self, left_line, right_line):
        pass

    def state_machine(self):
        # State 1: lane keeping + pedestrian search
        # State 2: pedestrian detection + stopping condition + free lane detection
        # State 3: lane keeping

        if self.state == 1:
            left_line, right_line = self.find_lane()
            self.waypoint = self.calculate_waypoint(left_line, right_line)
            if self.detect_pedestrian():
                self.state = 2
        elif self.state == 2:
            if self.detect_pedestrian():
                self.waypoint = (self.width // 2, self.height // 2)  # Stop
            else:
                self.state = 3
        elif self.state == 3:
            left_line, right_line = self.find_lane()
            self.waypoint = self.calculate_waypoint(left_line, right_line)

        self.update_mask()

    def update_mask(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.final_mask = mask

    def main(self):
        cap = cv2.VideoCapture("data/obstructed_pedestrian.mp4")
        self.hsv_obj = hsv("data/obstructed_pedestrian.mp4")

        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()

                cv2.imshow("Obstructed Pedestrian Detection", self.final_mask)
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

        cv2.imshow("Obstructed Pedestrian Detection", self.final_mask)

        return self.final_mask, self.waypoint
        