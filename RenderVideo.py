#!/usr/bin/python
import sys
import numpy as np
import cv2
from CameraCalibration import CalibrateCamera
from ProcessImage import ProcessImage
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
    clip1 = VideoFileClip('source_videos/' + filename)#.subclip(0,5)

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
    ld = ProcessImage()
    ld.fit()

    if False:
        image = mpimg.imread('test.jpg')
        ld.process_image(image)

        return
    else:
        white_clip = clip1.fl_image(ld.process_image) # color images

        white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()
