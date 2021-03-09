# Pedestrian-detection-and-tracking
The program detects a random pedestrian on a given video, then the program detects the same pedestrian on every frame. written in Python using OpenCV library.

There are two versions:
The first one is using a feature points matching to detect the same pedestrian on every frame. This version can be view in annotaion.py file.
The second version is using the MCRT tracker that OpenCV offer, the detects points from the Hog Descriptor are given to the tracker. When the pedestrian is out of the frame, the  tracker is initialized to new pedestrian. This version can be viewed in tracker.py file.

A sample output video can be viewed in output.mp4 file.

