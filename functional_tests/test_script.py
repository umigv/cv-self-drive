from pedestrian_lane_changing import LaneChangeStateMachine
from obstructed_pedestrian_detection import ObstructedPedestrianDetection
from curved_lane_keeping import CurvedLanekeeping
# from tire_detection_img import TireDetection  # Need to po tire detection to use the general functional test scaffold 

# ===================================================
# Specify Mode for which test you would like to run: 
# mode = "pedestrian-lane-change"
mode = "obstructed-pedestrian"
# mode = "curved-lane"
# mode = "tire-detection"

# ===================================================
# Specify what the test path is (what video you are using to test)
# if nothing it will try to use webcam / cam @ index 0

pedestrian_lane_change_test_path = "../data/HD2K_SN36466710_18-50-10.mp4" 
# obstructed_pedestrian_test_path = "../data"
# curved_lane_test = "../data"
functional_test_path = "" 


# ===================================================




if(mode ==  "pedestrian-lane-change"):
    pedestrian_lane_change_test = LaneChangeStateMachine(pedestrian_lane_change_test_path) # Change test path depending on the test that is being run
    pedestrian_lane_change_test.set_right_to_left()
    pedestrian_lane_change_test.run()

if(mode == "curved-lane"):
    curved_lane_test = CurvedLanekeeping()

    
if(mode == "obstructed-pedestrian"):
    obstructed_pedestrian_test = ObstructedPedestrianDetection()\

if(mode == "tire-detection"):
    # tire_detection_test = 
    x = 0
