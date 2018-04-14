import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os.path
import glob
from CameraCalibration import CalibrateCamera
from ImageSegmentation import abs_sobel_thresh, mag_thresh, dir_threshold, color_segmentation, \
                            mask_region_of_interest, img2gray
from LaneFit import LaneFit
from moviepy.editor import VideoFileClip
from time import time
import logging

class ProcessImageLane():
    def __init__(self):
        self.shape = None
        self.roi = None
        self.calCam = None
        self.laneFit = None
        self.image_cnt = 0
        self.DEBUG_IMAGE = True
        self.DEBUG_IMAGE_FOLDER = 'challenge_debug'
        self.DEBUG_VIDEO = False
        self.segmentation = 1

    def fit(self, shape, M, MInverse, roi=None, calCam=None):
        """
        First call before processing images
        """
        self.shape = shape

        # region of interest
        if roi != None:
            self.roi = roi
        else:
            self.roi = { 'bottom_y':       shape[0],
                        'bottom_x_left':  int(shape[1]*0.05),
                        'bottom_x_right': int(shape[1]*0.95),
                        'top_y':          int(shape[0]*0.6),
                        'top_x_left':     int(shape[1]*0.45),
                        'top_x_right':    int(shape[1]*0.55),            
                        }
            self.roi.update({'top_center': int((self.roi['top_x_left'] + self.roi['top_x_right']) / 2)})

        if calCam is not None:
            self.calCam = calCam
        else:
            self.calCam = CalibrateCamera.load()
        
            if self.calCam is None:
                images = glob.glob('camera_cal/calibration*.jpg')
                self.calCam = CalibrateCamera()
                self.calCam.findCorners(images, (9, 6))
                self.calCam.calibrateCamera()
                self.calCam.write()

        self.laneFit = LaneFit()

        self.M = M
        self.MInverse = MInverse
        self.frame = 0
        self.update = False

    def writeInfo(self, img, laneparams):
        """
        Writes information to the image
        """
        # empty image
        box_img = np.zeros_like(img).astype(np.uint8)

        # draw rectangle and arrow for position deviation
        box_img = cv2.rectangle(box_img, (10, 10), (int(1280/2-10), 150), (0, 0, 100), thickness=cv2.FILLED)
        box_img = cv2.rectangle(box_img, (int(1280/2-10), 10), (1280-10, 150), (0, 0, 100), thickness=cv2.FILLED)
        box_img = cv2.arrowedLine(box_img, (500, 60), (int(500 + laneparams['middle_phys'] * 200), 60), (255,0,0), 5)
        
        img = cv2.addWeighted(img, 1.0, box_img, 1.0, 0.)

        font = cv2.FONT_HERSHEY_SIMPLEX
        pos_str = 'Left curve:  {:06.0f}'.format(laneparams['left_curverad'])
        pos_str2 = 'Right curve: {:06.0f}'.format(laneparams['right_curverad'])
        pos_str3 = 'Middle: {:.2f}m'.format(laneparams['middle_phys'])
        frame_str = 'Frame: {}'.format(self.frame)

        left_pos = 40
        top_pos = 40
        delta_height = 30
        # write text to image
        cv2.putText(img, pos_str ,(left_pos, top_pos), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, pos_str2 ,(left_pos, top_pos+delta_height), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, pos_str3 ,(400, top_pos), font, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, frame_str ,(left_pos, top_pos+3*delta_height), font, 0.8, (255,255,255), 1, cv2.LINE_AA)

        return img

    def process_image(self, img_orig):
        """
        Processes an image
        """
        t0 = time()
        img = self.calCam.undistort(img_orig)

        if self.DEBUG_IMAGE:
            img_savename = 'image{0:04d}.jpg'.format(self.image_cnt)
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/original/'+img_savename, cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/undist/'+img_savename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_undist = img
        ksize = 9

        # gray conversion
        gray = img2gray(img)

        # Apply each of the thresholding functions
        if self.segmentation == 0:
            # magnitude and direction of edges, color
            mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 255))
            dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(np.pi/4*0.9, np.pi/4*1.5))
            color_seg = color_segmentation(img)

            seg_img_raw = color_seg & mag_binary & dir_binary
        
        else:
            # color segmentation and canny edge detection
            color_seg = color_segmentation(img)
            
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
            canny = cv2.Canny(blur_gray, 40, 80).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            # dilate canny edges which are very sharp
            canny = cv2.dilate(canny, kernel, iterations = 2)
            
            # segmented image with color and canny
            seg_img_raw = (color_seg & canny)

            if self.DEBUG_IMAGE:
                cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/seg_color/'+img_savename, color_seg.astype(np.uint8) * 255)
                cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/seg_canny/'+img_savename, canny.astype(np.uint8) * 255)
                cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/seg_comb/'+img_savename, seg_img_raw.astype(np.uint8) * 255)
        
        seg_img = seg_img_raw.astype(np.uint8) * 255

        seg_img[690:,:] = 0

        # mask image
        # region of interest not used
        #seg_img, vertices = mask_region_of_interest(seg_img, self.roi)
        seg_img = np.dstack((seg_img, seg_img, seg_img))

        #visualization = np.dstack((np.zeros(dir_binary.shape), (dir_binary & mag_binary), color_seg)).astype(np.uint8) * 255
        
        # warp to birds eye perspective
        seg_img_warped = cv2.warpPerspective(seg_img, self.M, (seg_img.shape[1], seg_img.shape[0]), flags=cv2.INTER_LINEAR)
        
        # thresholding for interpolated pixels
        seg_img_warped = (seg_img_warped > 100).astype(np.uint8) * 255

        if self.DEBUG_IMAGE:
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/seg_warped/'+img_savename, seg_img_warped)
            #undist_img_warped = cv2.warpPerspective(img_undist, self.M, (img_undist.shape[1], img_undist.shape[0]), flags=cv2.INTER_LINEAR)
            #cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/undist_warped/'+img_savename, undist_img_warped)
        
        # do LaneFit algorithm on image
        laneparams = self.laneFit.procVideoImg(seg_img_warped, margin=60, numwin=20)

        lane_img = laneparams['img']

        if self.DEBUG_IMAGE:
            comb = cv2.addWeighted(seg_img_warped, 0.5, lane_img, 0.8, 0)
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/combined/'+img_savename, cv2.cvtColor(comb, cv2.COLOR_BGR2RGB))

        lane_img_unwarped = cv2.warpPerspective(lane_img, self.MInverse, (lane_img.shape[1], lane_img.shape[0]), flags=cv2.INTER_LINEAR)
        
        # combine original image with detection
        img = cv2.addWeighted(img, 1, lane_img_unwarped, 0.7, 0)
        
        # time measurement
        t1 = time()        
        logging.info('process_image: runtime = ' + str(t1-t0))

        # debug video with different segmentations used in this algorithm
        if self.DEBUG_VIDEO:
            img = np.dstack((np.zeros_like(seg_img_raw), canny, color_seg)).astype(np.uint8) * 255
            img = cv2.addWeighted(img, 1, lane_img_unwarped, 0.9, 0)

        if self.DEBUG_IMAGE:
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/result/'+img_savename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.image_cnt += 1     # for number of debug image

        undist_img_warped = cv2.warpPerspective(img_undist, self.M, (img_undist.shape[1], img_undist.shape[0]), flags=cv2.INTER_LINEAR)

        # add smaller images of the segmented and warped image
        y_offset = 0
        output_size = (720, 1280, 3)
        ret_img = np.zeros(output_size).astype(np.uint8)

        warped_smaller = cv2.resize(undist_img_warped, (0, 0), fx=0.15, fy=0.15)
        warped_semented_smaller = cv2.resize(comb, (0, 0), fx=0.15, fy=0.15)
        warped_combined_smaller = cv2.addWeighted(warped_smaller, 1, warped_semented_smaller, 0.7, 0)

        color_smaller = cv2.resize(color_seg.astype(np.uint8), (0, 0), fx=0.15, fy=0.15).reshape((108, 192, 1)) * 255
        canny_smaller = cv2.resize(canny.astype(np.uint8), (0, 0), fx=0.15, fy=0.15).reshape((108, 192, 1)) * 255
        combined_smaller = cv2.resize((color_seg & canny).astype(np.uint8), (0, 0), fx=0.15, fy=0.15).reshape((108, 192, 1)) * 255
        stacked_smaller = np.dstack((combined_smaller, canny_smaller, color_smaller)).astype(np.uint8)

        ret_img[y_offset:img.shape[0]+y_offset, :img.shape[1], :] = img

        # write information to image
        ret_img = self.writeInfo(ret_img, laneparams)

        offset_small = 26
        offset_x_small = 684
        ret_img[offset_small:offset_small + 108, offset_x_small:offset_x_small + 192, :] = stacked_smaller
        ret_img[offset_small:offset_small + 108, offset_x_small+200:offset_x_small+392, :] = warped_combined_smaller

        self.frame += 1

        return ret_img

def main():
    white_output = 'processed_videos/challenge_video.mp4'

    clip1 = VideoFileClip("source_videos/challenge_video.mp4").subclip(0,5)

    target_left_x = 300
    target_right_x = 1002
    target_top_y = 0
    target_bottom_y =690
    src_points = np.float32([[283, 664], [548, 480], [736, 480],  [1019, 664]])
    dst_points = np.float32([[target_left_x, target_bottom_y], [target_left_x, target_top_y],
                                [target_right_x, target_top_y], [target_right_x, target_bottom_y]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Mi = cv2.getPerspectiveTransform(dst_points, src_points)

    ld = LaneDetect()
    ld.fit((720, 1280), M, Mi)

    white_clip = clip1.fl_image(ld.process_image) # color images

    white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()
