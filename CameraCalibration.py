import numpy as np
import cv2
import glob
import pickle
import os.path
import logging
import matplotlib.pyplot as plt

class CalibrateCamera():

    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.img_size = (0, 0)
        self.corner_size = (0, 0)
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def findCorners(self, images, corner_size):
        self.corner_size = corner_size
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.corner_size[0] * self.corner_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.corner_size[0], 0:self.corner_size[1]].T.reshape(-1,2)
        
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.img_size = (img.shape[1], img.shape[0])

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.corner_size, None)

            # If found, add object points, image points
            if ret:      
                print("Found corners in " + fname)      
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                #debug_img = cv2.drawChessboardCorners(img, corner_size, corners, ret)
                #cv2.imwrite('debug_images/' + fname, debug_img)



    def calibrateCamera(self):
        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)

        return self.mtx, self.dist

    def undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return undist

    def write(self, filename='obj_save/CameraCalibration.p'):
        directory = os.path.dirname(filename)

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(filename, 'wb') as pf:
            pickle.dump(self, pf)

    @staticmethod
    def load(filename='obj_save/CameraCalibration.p'):
        loaded = None
        if os.path.isfile(filename):
            with open(filename, 'rb') as pf:
                logging.info('Loading CalibrateCamera from ' + filename)
                loaded = pickle.load(pf)

        return loaded


def main():
    logging.basicConfig(level=logging.DEBUG)

    # load precalulated calibration data if available
    calCam = CalibrateCamera.load()

    # if no precalculated data is available start calculation
    if True: #calCam == None:
        images = glob.glob('camera_cal/calibration*.jpg')

        calCam = CalibrateCamera()

        calCam.findCorners(images, (9, 6))

        calCam.calibrateCamera()

        calCam.write()

    # for debuggind and documentation    
    debug_img = cv2.imread('camera_cal/calibration2.jpg')
    debug_img_undist = calCam.undistort(debug_img)
    cv2.imwrite('debug_images/camera_cal/calibration2_undist.jpg', debug_img_undist)
    
    print(calCam.mtx)
    print(calCam.dist)

if __name__ == "__main__":
    main()