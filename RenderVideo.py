#!/usr/bin/python
import sys
import numpy as np
import cv2
from CameraCalibration import CalibrateCamera
from ProcessImage import ProcessImage
from ProcessImageLane import ProcessImageLane
from moviepy.editor import VideoFileClip
import glob
import matplotlib.image as mpimg

def main():
    if (len(sys.argv) > 1) and isinstance(sys.argv[1], str):
        filename = sys.argv[1]
    else:
        filename = 'test_video.mp4'
    
    print('Processing file ' + filename)

    white_output = 'processed_videos/' + filename
    clip1 = VideoFileClip('source_videos/' + filename)#.subclip(27, 30)

    #################################
    # Lane detection
    #################################

    # calculate matrices for perspective transformation
    target_left_x = 300
    target_right_x = 1002
    target_top_y = 0
    target_bottom_y =690
    src_points = np.float32([[283, 664], [548, 480], [736, 480],  [1019, 664]])
    dst_points = np.float32([[target_left_x, target_bottom_y], [target_left_x, target_top_y],
                                [target_right_x, target_top_y], [target_right_x, target_bottom_y]])

    # transformation to bird's eye view
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # transformation back to normal view
    Mi = cv2.getPerspectiveTransform(dst_points, src_points)

    # calculate or load camera calibration
    calCam = CalibrateCamera.load()

    if calCam == None:
        images = glob.glob('camera_cal/calibration*.jpg')
        calCam = CalibrateCamera()
        calCam.findCorners(images, (9, 6))
        calCam.calibrateCamera()
        calCam.write()

    # class which will process the images, initialize with image size and
    # transformation matrices
    ld = ProcessImageLane()
    ld.fit((720, 1280), M, Mi, calCam=calCam)

    #################################
    # Vehicle detection
    #################################

    # class to detect the vehicles
    # for a quick merge with the lane detection the lane detection pipeline is added
    vd = ProcessImage()
    vd.fit(ld)

    if False:
        image = mpimg.imread('test.jpg')
        vd.process_image(image)

        return
    else:
        white_clip = clip1.fl_image(vd.process_image) # color images

        white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()
