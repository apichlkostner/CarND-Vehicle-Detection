import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from pathlib import Path
from sklearn.svm import LinearSVC, SVC
import sklearn.svm as svm
import sklearn.grid_search as grid_search
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from skimage import data, exposure
import pickle
from HelperFunctions import *


class FindCars():
    def __init__(self):
        self.colorTransform = cv2.COLOR_RGB2YCrCb
        self.clf = None
        self.x_scaler = None
        self.frame_nr = 0
        self.debug_folder = ''
        self.use_spatial_features = None
        self.use_color_features = None
        self.use_hog_features = None
        self.proba = None
        self.pix_per_cell = None
        self.cell_per_block = None
        self.spatial_size = None
        self.hist_bins = None
        self.orient = None

    def fit(self, swd):
        self.clf = swd['clf']
        self.x_scaler = swd['x_scaler']
        self.frame_nr = 0
        self.debug_folder = ''
        self.use_spatial_features = swd['use_spatial']
        self.use_color_features = swd['use_color']
        self.use_hog_features = swd['use_hog']
        self.proba = swd['proba']
        self.pix_per_cell = swd['pix_per_cell']
        self.cell_per_block = swd['cell_per_block']
        self.spatial_size = swd['spatial_size']
        self.hist_bins = swd['hist_bins']
        self.orient = swd['orient']

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, swd):
        
        ystart = swd['y_top']
        ystop  = swd['y_bottom']
        xstart = swd['x_left']
        scale  = swd['scale']

        # region of interest
        img_tosearch = img[ystart:ystop, xstart:, :]

        # color transformation
        ctrans_tosearch = cv2.cvtColor(img_tosearch, self.colorTransform)

        # resize image if scale != 1
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale),
                                                           np.int(imshape[0] / scale)))
            
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient * self.cell_per_block ** 2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        t0 = time.time()

        if False: #True and (scale == 1):
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            #ch1 = clahe.apply(ch1)
            hog1, img1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
            hog2, img2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
            hog3, img3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
            if False:
                #img1 = exposure.rescale_intensity(img1, in_range=(0, 10))
                #print('Shape = {} min = {}  max = {}'.format(img1.shape, img1.min(), img1.max()))
                #cv2.imwrite(folder + 'hog/frame_hog1_{:04d}.jpg'.format(frame_nr), img1)
                print('{} {} {}'.format(self.orient, self.pix_per_cell, self.cell_per_block))
                #fd, hog_image = hog(ch3, orientations=8, pixels_per_cell=(16, 16),
                #            cells_per_block=(1, 1), visualise=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                ax1.axis('off')
                ax1.imshow(ch3, cmap=plt.cm.gray)
                ax1.set_title('Input image')

                # Rescale histogram for better display
                hog_image_rescaled = exposure.rescale_intensity(img3, in_range=(0, 10))

                ax2.axis('off')
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                plt.show()

            if True:
                #print('Min {}  max {}  mean {}'.format(img1.min(), img1.max(), img1.mean()))
                img1 = (img1 / img1.max() * 255).astype(np.uint8) 
                img2 = (img2 / img2.max() * 255).astype(np.uint8) 
                img3 = (img3 / img3.max() * 255).astype(np.uint8) 
                cv2.imwrite(self.folder + 'hog/frame_hog1_{:04d}.jpg'.format(self.frame_nr), img1)
                cv2.imwrite(self.folder + 'hog/frame_hog2_{:04d}.jpg'.format(self.frame_nr), img2)
                cv2.imwrite(self.folder + 'hog/frame_hog3_{:04d}.jpg'.format(self.frame_nr), img3)
        else:
            img1 = []
            hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        t1 = time.time()

        boxes = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                if self.use_hog_features:
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step

                    features = []

                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    features.append(hog_features)

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]

                if (subimg.shape[0] != 64) or (subimg.shape[1] != 64):
                    print("Error: shape = {}".format(subimg.shape))
            
                
                if self.use_spatial_features:
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    features.append(spatial_features)

                if self.use_color_features:
                    hist_features = color_hist(subimg, nbins=self.hist_bins)
                    features.append(hist_features)

                # Scale features and make a prediction
                test_features = self.x_scaler.transform(np.hstack(features).reshape(1, -1))
                
                if self.proba:
                    # use probalibilies
                    #print('Use proba = {}'.format(self.clf.predict_proba(test_features)))
                    proba = self.clf.predict_proba(test_features)
                    test_prediction = 1 * (proba[0, 1] > 0.95)
                else:
                    # just make prediction
                    test_prediction = self.clf.predict(test_features)
                #print('Prob = {}, Pred = {}'.format(proba, test_prediction2))
                
                if test_prediction == 1:
                    xbox_left = np.int((xleft)*scale + xstart)
                    ytop_draw = np.int((ytop)*scale + ystart)
                    win_draw = np.int(window*scale)
                    boxes.append([[xbox_left, ytop_draw], [xbox_left+win_draw, ytop_draw+win_draw]])
        
        t2 = time.time()

        print('{} windows with scale {} needed time {}'.format(nxsteps * nysteps, scale, t2 - t1))

        return boxes, img1