import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from pathlib import Path
from sklearn.svm import LinearSVC, SVC
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from skimage import data, exposure
import pickle
#from skimage.feature import hog
from HelperFunctions import *
from FindCars import *
from Segmentation import *
from Model import *
from FeatureExtract import FeatureExtractor

from sklearn.model_selection import train_test_split
import concurrent.futures

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows2(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 1.), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        #print('Feature 1 min={}  max={}'.format(features.min(), features.max()))
        features[np.isnan(features)] = 0.0
        #print('Feature 2 min={}  max={}'.format(features.min(), features.max()))
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def search_windows(img, windows, conf, feat_extr):
    scaler = conf['x_scaler']
    clf = conf['model']
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        if img.shape[0] != 64:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = feat_extr.calc_features(test_img)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        if True:
            proba = clf.predict_proba(test_features)
            prediction = 1 * (proba[0, 1] > 0.9)
        else:
            prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)

    #8) Return windows for positive detections
    return on_windows

class ProcessImage():
    def __init__(self):
        self.model_config = {'color_space': cv2.COLOR_RGB2YCrCb, 'orient': 9,
                             'pix_per_cell': 16, 'cell_per_block': 2,
                             'hog_channel': 'ALL', 'spatial_size': (16, 16),
                             'hist_bins': 16, 'spatial_feat': True,
                             'hist_feat': True, 'hog_feat': True, 'probability': True}
        self.y_start_stop = [350, 650] # Min and max in y to search in slide_window()
        self.x_start_stop = [640, 1280]
        self.heat = None
        self.thres_cnt = 0
        self.cnt = 0
        self.DEBUG = True
        self.frame_nr = 0
        self.use_sliding_window = True
        self.parallel = 'serial' #'process'

    def fit(self):
        model = Model()
        model_map = model.fit(self.model_config)

        self.model_config = model_map['model_config']
        self.model_config['model'] = model_map['model']
        self.model_config['x_scaler'] = model_map['x_scaler']
        self.feat_extr = FeatureExtractor(self.model_config)
        self.model_config['feat_extr'] = self.feat_extr
        
        self.find_cars = FindCars()
        self.find_cars.fit(self.model_config)

        # building lists with windows for sliding window algorithm
        # needs 1ms
        self.windows = []
        self.windows.append(slide_window(x_start_stop=(800, 1280), y_start_stop=(390, 518), 
                                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)))
        self.windows.append(slide_window(x_start_stop=(800, 1280), y_start_stop=(390, 518), 
                                    xy_window=(96, 96), xy_overlap=(0.5, 0.5)))
        # self.windows.append(slide_window(x_start_stop=(1050, 1280), y_start_stop=(550, 700), 
        #                             xy_window=(128, 128), xy_overlap=(0.75, 0.75)))
        

    def process_image(self, img):
        if self.DEBUG:
            folder = 'debug/project_video/'
            #folder = 'debug/test_video/'
            if self.frame_nr < 0:
                self.frame_nr += 1
                return img

        # save original image in RGB for drawing results
        draw_image = np.copy(img)
        
        # transform to selected colorspace of model
        img = cv2.cvtColor(img, self.model_config['color_space'])

        if self.heat is None:
            self.heat = np.zeros_like(img[:,:,0]).astype(np.float)
        
        if self.use_sliding_window:
            if self.parallel == 'process':
                #t0 = time.time()
                box_list = []
                sliding_window_desc = [
                    (img, self.windows[0], self.model_config, self.feat_extr),
                    (img, self.windows[1], self.model_config, self.feat_extr),
                    (img, self.windows[2], self.model_config, self.feat_extr)]
                
                if self.parallel == 'process':
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        res = executor.map(find_cars_sliding, sliding_window_desc)
                        for r in res:
                            print('Adding {} boxes'.format(len(r)))
                            box_list.extend(r)
                #t1 = time.time()
                #print('Time for alls scales (parallel): {:.3f}'.format(t1 - t0))                
            else:
                #t0 = time.time()
                box_list = []
                for windows in self.windows:
                    box_list.extend(search_windows(img, windows, self.model_config, self.feat_extr))
                
                #t1 = time.time()
                #print('Time for alls scales (serial): {:.3f}'.format(t1 - t0))
        else:
            box_list = []
            sliding_window_desc = [(img, {'scale': 0.7, 'y_top': 400, 'y_bottom': 550, 'x_left': 640, 'x_right': 1280}),
                                   (img, {'scale': 1.0, 'y_top': 400, 'y_bottom': 600, 'x_left': 640, 'x_right': 1280}),
                                   #(img, {'scale': 1.5, 'y_top': 400, 'y_bottom': 650, 'x_left': 640, 'x_right': 1280}),
                                   #(img, {'scale': 2.0, 'y_top': 400, 'y_bottom': 650, 'x_left': 640, 'x_right': 1280}),
                                    ]
            t0 = time.time()
            if self.parallel == 'process':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    res = executor.map(self.find_cars.find_cars, sliding_window_desc)
                    for r in res:
                        box_list.extend(r[0])
            else:
                for i, swd in enumerate(sliding_window_desc):                    
                    box, _ = self.find_cars.find_cars(swd)
                    box_list.extend(box)
                
            t1 = time.time()
            print('Time for alls scales: {:.3f}'.format(t1 - t0))

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, box_list)

        #print('Min {} Max {}'.format(heat.min(), heat.max()))
        heat = np.clip(heat, 0, 3)
        #heat = apply_threshold(heat, 2)

        #print('Min {} Max {}'.format(heat.min(), heat.max()))

        alpha = 0.8
        #self.heat = alpha * self.heat + (1. -alpha) * heat
        self.heat = alpha * self.heat + heat
        self.heat = np.clip(self.heat, 0, 6)
        #print('Min {} Max {}'.format(self.heat.min(), self.heat.max()))
        
        heat = self.heat.copy()
        heat = apply_threshold(heat, 5)

        #print('Min {} Max {}'.format(heat.min(), heat.max()))

        heatmap_img = np.dstack((heat, np.zeros_like(heat), np.zeros_like(heat)))

        #hog_image = np.zeros_like(draw_image)
        #hog_image[self.y_start_stop[0]:self.y_start_stop[1], self.x_start_stop[0]:self.x_start_stop[1], 2] = hog_img
        #hog_image[self.y_start_stop[0]:self.y_start_stop[1], self.x_start_stop[0]:self.x_start_stop[1], 1] = hog_img
        #hog_img = np.dstack((np.zeros_like(hog_img), np.zeros_like(hog_img), hog_img))
        #hog_img = cv2.resize(hog_img, (draw_image.shape[1], draw_image.shape[0]))
        #draw_image = cv2.addWeighted(draw_image, 0.1, hog_image, 1., 0)
        

        # Find final boxes from heatmap using label function
        labels = label(heat)

        window_img, boxes = draw_labeled_bboxes(draw_image, labels)

        # Try flood fill inside bounding boxes for better segmentation
        if False:
            ff = FloodFill()
            window_img = ff.fill(window_img, boxes)
        
        # debug: show single boxes and heatmap
        if False:
            window_img = draw_boxes(window_img, box_list, color=(0, 255, 0), thick=1)
            draw_image = cv2.addWeighted(draw_image, 1., 25*heatmap_img.astype(np.uint8), 1., 0)


        if self.DEBUG:
            #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            window_img_bgr = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(folder + 'original/frame_{:04d}.jpg'.format(self.frame_nr), img_bgr)
            cv2.imwrite(folder + 'processed/frame_{:04d}.jpg'.format(self.frame_nr), window_img_bgr)
            self.frame_nr += 1

        return window_img    



