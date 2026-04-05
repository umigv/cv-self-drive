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

    def state_machine(self):
        # State 1: lane keeping + pedestrian search
        # State 2: pedestrian detection + stopping condition
        # State 3: lane keeping




    def update_mask(self):
        # Placeholder for mask update logic
        pass

    def main(self):
        cap = cv2.VideoCapture("data/obstructed_pedestrian.mp4")
        self.hsv_obj = hsv("data/obstructed_pedestrian.mp4")

        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()
        