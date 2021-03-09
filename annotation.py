# ;==========================================
# ; Title:  Annotate a Pedestrian in a video
# ; Author: Anna Mi
# ; Date:   27 Feb 2021
# ;==========================================
# Annotate a Pedestrian in a video:
# Given a video that includes pedestrians, you need to mark one person on a frame using a bounding
# rectangle. Then you program will extract feature points and locate the same person on the next
# frames. You need to allow the use to correct the location and size of the detected person on any
# frame.
# ;==========================================
# Steps:
# detect a pedestrian in video - using hog detector
# extract feature points (read corners) - using sparse optical flow
# compare the previous and current frames, and adjust the bounding box according to the points' locations
# ;==========================================
# Notes - it is possible to adjust the feature_params for the ShiTomasi corner detection,
# it can improve the quality of the pedestrian tracking, depending on the video.


import cv2
import numpy as np
import sys
import imutils


def initial_pedestrian_detection(image):
    """
    Detecting a pedestrian in the initial frame (or if there are no feature points tracked in some frame)
    :param image: the current frame image
    :return: (x, y, w, h) that defines the area in which a pedestrian was detected
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detecting the pedestrian regions in the image
    (regions, _) = hog.detectMultiScale(image,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)
    # selecting a pedestrian to track in the next frames
    print("initial_pedestrian_detection")
    return regions[0]


def filter_matches(matches):
    # filter matches-Lowe's ratio test
    ratio_thresh = 0.7
    temp_matches = []
    good_matches = []
    for list in matches:
        if len(list) > 1:
            temp_matches.append(list)
    for m, n in temp_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches


def update_bounding_box_width_old(src_pts, dst_pts, w, previous_proportion):
    # calculate the distance difference between feature points and decide if adjustment is needed.
    updated_w = 0
    width_pedestrian = 0
    width_frame = 0

    # find leftmost and rightmost point values in the pedestrian image, and calculate the distance between (width)
    rightmost_pt_pedestrian = 0
    leftmost_pt_pedestrian = 400
    for [i_point] in src_pts:
        if i_point[0] < leftmost_pt_pedestrian:
            leftmost_pt_pedestrian = i_point[0]
        if i_point[0] > rightmost_pt_pedestrian:
            rightmost_pt_pedestrian = i_point[0]
    width_pedestrian = rightmost_pt_pedestrian - leftmost_pt_pedestrian

    # same process for the frame
    rightmost_pt_frame = 0
    leftmost_pt_frame = 400
    for [i_point] in dst_pts:
        if i_point[0] < leftmost_pt_frame:
            leftmost_pt_frame = i_point[0]
        if i_point[0] > rightmost_pt_frame:
            rightmost_pt_frame = i_point[0]
    width_frame = rightmost_pt_frame - leftmost_pt_frame

    # check if the proportions have changed and an adjustment is needed
    # current_proportion = width_frame / width_pedestrian
    if previous_proportion == 0:
        previous_proportion = width_pedestrian / w
    current_proportion = width_frame / w

    if current_proportion == previous_proportion:
        updated_w = w
    else:
        updated_w = int(width_frame / previous_proportion)
        if updated_w not in range(w - 10, w + 10):
            updated_w = w

    previous_proportion = current_proportion

    return int(updated_w), previous_proportion


def update_bounding_box_height_old(src_pts, dst_pts, h, previous_proportion):
    # calculate the distance difference between feature points and decide if adjustment is needed.
    updated_h = 0
    height_pedestrian = 0
    height_frame = 0

    # find leftmost and rightmost point values in the pedestrian image, and calculate the distance between (width)
    bottommost_pt_pedestrian = 0
    upmost_pt_pedestrian = 400
    for [i_point] in src_pts:
        if i_point[1] < upmost_pt_pedestrian:
            upmost_pt_pedestrian = i_point[1]
        if i_point[1] > bottommost_pt_pedestrian:
            bottommost_pt_pedestrian = i_point[1]
    height_pedestrian = bottommost_pt_pedestrian - upmost_pt_pedestrian

    # same process for the frame
    bottommost_pt_frame = 0
    upmost_pt_frame = 400
    for [i_point] in dst_pts:
        if i_point[1] < upmost_pt_frame:
            upmost_pt_frame = i_point[1]
        if i_point[1] > bottommost_pt_frame:
            bottommost_pt_frame = i_point[1]
    height_frame = bottommost_pt_frame - upmost_pt_frame

    # check if the proportions have changed and an adjustment is needed
    if previous_proportion == 0:
        previous_proportion = height_pedestrian / h
    current_proportion = height_frame / h

    if current_proportion == previous_proportion:
        updated_h = h
    else:
        updated_h = int(height_frame / previous_proportion)
        if updated_h not in range(h - 10, h + 10):
            updated_h = h

    previous_proportion = current_proportion

    return int(updated_h), previous_proportion


def update_bounding_box_corner_pt_old(src_pts, dst_pts, x, y):
    x_mean = 0
    y_mean = 0
    x_mean_frame = 0
    y_mean_frame = 0

    # find mean feature points' location on x and y axis - on pedestrian sub image
    for [i_point] in src_pts:
        x_mean += i_point[0]
        y_mean += i_point[1]
    x_dif_pedestrian = int(x_mean / len(src_pts))
    y_dif_pedestrian = int(y_mean / len(src_pts))

    # find mean feature points' location on x and y axis - on whole frame image
    for [i_point] in dst_pts:
        x_mean_frame += i_point[0]
        y_mean_frame += i_point[1]
    x_mean_frame = int(x_mean_frame / len(dst_pts))
    y_mean_frame = int(y_mean_frame / len(dst_pts))
    x_dif_frame = x_mean_frame - x
    y_dif_frame = y_mean_frame - y

    # decide whether to update the corner point or not
    print(x_dif_frame - x_dif_pedestrian)
    if x_dif_frame - x_dif_pedestrian in range(-10, 10):
        if x_dif_pedestrian - x < x_dif_frame - x:
            x += x_dif_frame - x_dif_pedestrian
        if x_dif_pedestrian - x > x_dif_frame - x:
            x -= x_dif_frame - x_dif_pedestrian

    print(y_dif_frame - y_dif_pedestrian)
    if y_dif_frame - y_dif_pedestrian in range(-10, 10):
        if y_dif_pedestrian - y < y_dif_frame - y:
            y += y_dif_frame - y_dif_pedestrian
        if y_dif_pedestrian - y > y_dif_frame - y:
            y -= y_dif_frame - y_dif_pedestrian

    return x, y


def check_pedestrian_left_the_frame(x, y, w, h, frame_w, frame_h):
    """
    Trying to indicate if the tracked pedestrian has left the frame
    :param x: x axis location
    :param y: y axis location
    :param w: width of the pedestrian's bounding box (from previous frame)
    :param h: height of the pedestrian's bounding box (from previous frame)
    :param frame_w: width of the frame
    :param frame_h: height of the frame
    :return: True if the pedestrian has left the frame, false otherwise
    """
    if x + w > frame_w + (frame_w / 2) or y + h > frame_h + (frame_h / 2):
        return True
    return False


def select_good_points(p0, p1, st, x, y, w, h):
    """
    Selecting good feature points to use for tracking the pedestrian
    :param p0: feature points from previous frame
    :param p1: feature points from current frame
    :param st: status array of each feature point (each item is set to 1 if the flow for the corresponding features has been found)
    :param x: x axis location
    :param y: y axis location
    :param w: width of the pedestrian's bounding box (from previous frame)
    :param h: height of the pedestrian's bounding box (from previous frame)
    :return: two arrays of old (from previous frame) and new (current frame) good points
    """
    # select good points
    good_new_temp = p1[st == 1]
    good_old_temp = p0[st == 1]

    good_new = []
    # select feature points of the selected pedestrian
    for pt in good_new_temp:
        if x < pt[0] < x + w and y < pt[1] < y + h:
            good_new.append(pt)
    good_old = []
    for pt in good_old_temp:
        if x < pt[0] < x + w and y < pt[1] < y + h:
            good_old.append(pt)

    good_new = np.array(good_new)
    good_old = np.array(good_old)

    return good_new, good_old


def adjust_bounding_box_corner_pt(good_new, good_old, x, y, h, w):
    """
    adjusts the corner point coordinates if needed
    :param good_new: good feature points from current frame
    :param good_old: good feature points from previous frame
    :param x: x axis location
    :param y: y axis location
    :param h: height of the pedestrian's bounding box (from previous frame)
    :param w: width of the pedestrian's bounding box (from previous frame)
    :return: updated values for (x,y) corner point
    """
    mean_h_new = 0
    mean_w_new = 0
    mean_h_old = 0
    mean_w_old = 0
    n = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        n += 1
        # a is the x/width coordinate, b is the y/height coordinate [same for c,d]
        a, b = new.ravel()
        c, d = old.ravel()

        # a = a + w
        # b = b + h
        # c = c + w
        # d = d + h
        # mean_h_new += a
        # mean_w_new += b
        # mean_h_old += c
        # mean_w_old += d
        mean_w_new += a
        mean_h_new += b
        mean_w_old += c
        mean_h_old += d

    # check if n (the number of good points) is not 0 yet, to avoid division by 0
    if n != 0:
        mean_h_new = int(mean_h_new / n)
        mean_w_new = int(mean_w_new / n)
        mean_h_old = int(mean_h_old / n)
        mean_w_old = int(mean_w_old / n)

        dif_h = mean_h_new - mean_h_old
        if dif_h != 0 and dif_h in range(-5, 5) and y + dif_h >= 0:
            y += dif_h
        dif_w = mean_w_new - mean_w_old
        if dif_w != 0 and dif_w in range(-5, 5) and x + dif_w >= 0:
            x += dif_w

    return x, y


def update_bounding_box_width(good_new, good_old, x, w):
    """
    adjusting the width of the bounding box if needed
    :param good_new: good feature points from current frame
    :param good_old: good feature points from previous frame
    :param x: x axis location
    :param w: width of the pedestrian's bounding box (from previous frame)
    :return: updated width value
    """
    # find leftmost and rightmost point values in the previous frame, and calculate the distance between them - width
    rightmost_pt_old = 0
    leftmost_pt_old = 400
    for i_point in good_old:
        if i_point[0] < leftmost_pt_old:
            leftmost_pt_old = i_point[0]
        if i_point[0] > rightmost_pt_old:
            rightmost_pt_old = i_point[0]
    width_old = round(rightmost_pt_old - leftmost_pt_old)

    # same process for the current frame
    rightmost_pt_new = 0
    leftmost_pt_new = 400
    for i_point in good_new:
        if i_point[0] < leftmost_pt_new:
            leftmost_pt_new = i_point[0]
        if i_point[0] > rightmost_pt_new:
            rightmost_pt_new = i_point[0]
    width_new = round(rightmost_pt_new - leftmost_pt_new)

    # the range is important - ensures that there are no large changes that create gaps between frames
    if width_old != width_new and width_old-width_new in range(-5, 5):
        x_addition = round((width_old - width_new) / 2)
        x += x_addition

        w_addition = width_new - width_old
        w += w_addition
    return w, x


def update_bounding_box_height(good_new, good_old, y, h):
    """
    adjusting the width of the bounding box if needed
    :param good_new: good feature points from current frame
    :param good_old: good feature points from previous frame
    :param y: y axis location
    :param h: height of the pedestrian's bounding box (from previous frame)
    :return: updated height value
    """
    # find bottommost and upmost point values in the previous frame, and calculate the distance between - height
    bottommost_pt_old = 0
    upmost_pt_old = 400
    for i_point in good_old:
        if i_point[1] < upmost_pt_old:
            upmost_pt_old = i_point[1]
        if i_point[1] > bottommost_pt_old:
            bottommost_pt_old = i_point[1]
    height_old = round(bottommost_pt_old - upmost_pt_old)

    # same process for the current frame
    bottommost_pt_new = 0
    upmost_pt_new = 400
    for i_point in good_new:
        if i_point[1] < upmost_pt_new:
            upmost_pt_new = i_point[1]
        if i_point[1] > bottommost_pt_new:
            bottommost_pt_new = i_point[1]
    height_new = round(bottommost_pt_new - upmost_pt_new)

    # the range is important - ensures that there are no large changes that create gaps between frames
    if height_old != height_new and height_old - height_new in range(-5, 5):
        y_addition = round((height_old - height_new) / 2)
        y += y_addition

        h_addition = height_new - height_old
        h += h_addition

    return h, y


def annotate_pedestrian(video_path):
    """
    primary function - detects a pedestrian and adjusts the bounding box in the following frames
    :return: displays the frames with a red bounding box which tracks a pedestrian
    """
    # cap = cv2.VideoCapture('pedestrians_1.mp4')
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    (x, y, w, h) = [0, 0, 0, 0]

    # prepare parameters for lucas kanade optical flow
    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while cap.isOpened():
        # reading the video stream
        ret, image = cap.read()

        if ret:
            image = imutils.resize(image, width=min(400, image.shape[1]))

            # no selected pedestrian from previous frame - find one
            if x == 0 and y == 0 and w == 0 and h == 0:
                (x, y, w, h) = initial_pedestrian_detection(image)

                out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1], image.shape[0]))

                # prepare params for ShiTomasi corner detection - ADJUST THE qualityLevel FOR BETTER RESULTS
                feature_params = dict(maxCorners=200,
                                      qualityLevel=0.1,
                                      minDistance=7,
                                      blockSize=7)

                # change the frame to gray - the feature points are better detected
                old_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # find initial feature points
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


            # a pedestrian was detected in the previous frame
            else:
                # change the frame to gray - the feature points are better detected
                frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                print('p0 len: ', len(p0))
                if len(p0) != 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # no feature points - meaning the pedestrian has left the frame or no feature points were discovered
                else:
                    # detect a new pedestrian to follow and find new feature points
                    (x, y, w, h) = initial_pedestrian_detection(image)
                    old_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # --------------select good points, relevant for the pedestrian--------------#
                good_new, good_old = select_good_points(p0, p1, st, x, y, w, h)

                # --------------adjust the width value if needed-----------------------------#
                w, x = update_bounding_box_width(good_new, good_old, x, w)

                # --------------adjust the height value if needed----------------------------#
                h, y = update_bounding_box_height(good_new, good_old, y, h)

                # --------------adjust the corner point (x,y) position-----------------------#
                x, y = adjust_bounding_box_corner_pt(good_new, good_old, x, y, h, w)

                # --------------update p0 for next frame handling----------------------------#
                p0 = good_new.reshape(-1, 1, 2)
                # old_gray = frame_gray.copy()
                # ---------------------------------------------------------------------------#



            # drawing bounding rectangle according to the detected region
            print('x:', x, 'y:', y, 'w:', w, 'h:', h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imwrite("current_frame.jpg", image)
            out.write(image)

            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    video_path = sys.argv[1]
    annotate_pedestrian(video_path)


if __name__ == '__main__':
    main()
