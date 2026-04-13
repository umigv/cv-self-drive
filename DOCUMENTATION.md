CV Self-Drive Documentation

## ```right_turn.py```

### To run on a video
Replace the filenames in these lines in `run(self)` with the filename of the video you want to run the algorithm on. To use a webcam, replace each filename with the number representing that webcam.
``` python
cap = cv2.VideoCapture("data/right_turn_cropped.mp4")
self.hsv_obj = hsv("data/right_turn_cropped.mp4")
```

### To run on a single frame
Call this function on your `RightTurn` object, with `hsv_indentifier` being the file to look at or the number of the camera to use (for HSV tuned values). `frame` is the OpenCV image frame to process.
``` python
def run_frame(self, hsv_indentifier, frame)
```

### Algorithm
* State 1: Drive forward, setting a constant waypoint straight ahead until we can no longer see the first yellow lane lines.
* State 2: Induce a turn to the right with a constant waypoint and guidelines until we can see the next set of yellow lane lines.
* State 3: Find the midpoint of the lane we need to enter and drive toward it until the waypoint becomes low enough.
* State 4: Look for a barrel, setting a waypoint at it. If we can't find a barrel, then we set the waypoint to the top of the lane.

## ```functional_tests/curved_lane_keeping.py```

### To run on a video
Replace the filenames in these lines in `run(self)` with the filename of the video you want to run the algorithm on. To use a webcam, replace each filename with the number representing that webcam.
``` python
cap = cv2.VideoCapture("data/left_curved_road.MOV")
self.hsv_obj = hsv("data/left_curved_road.MOV", barrel_mode = self.barrel_mode)
```

### To run on a single frame
Call this function on your `CurvedLanekeeping` object, with `hsv_indentifier` being the file to look at or the number of the camera to use (for HSV tuned values). `frame` is the OpenCV image frame to process.
``` python
def run_frame(self, hsv_indentifier, frame)
```

### Algorithm
* Look for a barrel. If we find one, set the waypoint on top of it.
* Otherwise, find the topmost point of each lane line within a particular search box, setting the waypoint to be the midpoint between the two points.

### Note about searchboxes:
* Keep the bounds symmetric, as this algorithm should be able to work when turning in either direction.