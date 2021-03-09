import sys
import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression


# The following program detect a random pedestrian using HOG descriptor
# The points that was detect are normalized based on the video style
# CSRT tracker is initialized with the points that was detected
# For every frame the tracker is updated if the object is still in the current frame
# If the object is out of the frame, we delete the current tracker, and initialize new one with new detect pedestrian
# The program are skipping 30 frames, if their isn't any detection
# The program finished and save the new video


def Detector(frame, HOGCV, resize):
    rects, weights = HOGCV.detectMultiScale(frame)
    rects = non_max_suppression(rects)

    if len(rects) != 0:
        x, y, w, h = rects[0]
        if resize is True:
            pass
        else:
            x = x + 20
            y = y + 20
            w = w - 40
            h = h - 40

        rects[0] = (x, y, w, h)
    return rects


def main(vid_name):
    tracking = True
    resize = True
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(vid_name)
    tracker = cv2.TrackerCSRT_create()
    ret, old_frame = cap.read()
    frame = imutils.resize(old_frame, width=min(400, old_frame.shape[1]))
    rects = Detector(frame, HOGCV, resize)
    if len(rects) == 0:
        resize = False
        rects = Detector(old_frame, HOGCV, resize)

    x, y, w, h = rects[0]
    # cv2.rectangle(frame,(x,y),(w,h),(255,255,255))
    # first_box = cv2.selectROI(frame, False)
    # ok = tracker.init(frame, first_box)
    if resize:
        ok = tracker.init(frame, (x, y, w, h))
    else:
        ok = tracker.init(old_frame, (x, y, w, h))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if resize:
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    else:
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (old_frame.shape[1], old_frame.shape[0]))
    while cap.isOpened():

        # work with countur, save it and then recatangle it

        ok, old_frame = cap.read()
        if ok:
            frame = imutils.resize(old_frame, width=min(400, old_frame.shape[1]))

            if tracking:
                if resize:
                    ok, first_box = tracker.update(frame)
                else:
                    ok, first_box = tracker.update(old_frame)

                if ok:
                    if resize:
                        x, y, _ = frame.shape
                    else:
                        x, y, _ = old_frame.shape
                    p1 = (int(first_box[0]), int(first_box[1]))
                    p2 = (int(first_box[0] + first_box[2]), int(first_box[1] + first_box[3]))

                    if ((p1[0] or p2[0]) >= y) or ((p1[0] or p2[0]) < 0) \
                            or ((p1[1] or p2[1]) >= x) or ((p1[1] or p2[1]) < 0):

                        tracking = False
                    else:
                        if resize:
                            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                        else:
                            cv2.rectangle(old_frame, p1, p2, (255, 0, 0), 2, 1)
                else:
                    tracking = False
            else:
                frame = imutils.resize(old_frame, width=min(400, old_frame.shape[1]))
                if resize:
                    rects = Detector(frame, HOGCV, resize)
                else:
                    rects = Detector(old_frame, HOGCV, resize)
                if len(rects) != 0:
                    del tracker
                    tracker = cv2.TrackerCSRT_create()

                    x, y, w, h = rects[0]

                    if resize:
                        ok = tracker.init(frame, (x, y, w, h))
                    else:
                        ok = tracker.init(old_frame, (x, y, w, h))
                    tracking = True
                else:
                    ind = cap.get(cv2.CAP_PROP_POS_FRAMES)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, ind + 30)

            if resize:
                out.write(frame)
                cv2.imshow("Tracking_r", frame)
            else:
                out.write(old_frame)
                cv2.imshow("Tracking", old_frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = sys.argv[1]
    main(path)
