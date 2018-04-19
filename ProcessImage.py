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

def search_windows(img, windows, conf, feat_extr):
    scaler = conf['x_scaler']
    clf = conf['model']
    
    on_windows = []
    # Iteration over all windows
    for window in windows:        
        if img.shape[0] != 64:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        
        # extract features
        features = feat_extr.calc_features(test_img)
        # scale features
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # using probabilites to reduce false positives
        if True:
            proba = clf.predict_proba(test_features)
            prediction = 1 * (proba[0, 1] > 0.9)
        else:
            prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)
    
    return on_windows

class ProcessImage():
    def __init__(self):
        self.model_config = {'color_space': cv2.COLOR_RGB2YCrCb, 'orient': 9,
                             'pix_per_cell': 16, 'cell_per_block': 2,
                             'hog_channel': 'ALL', 'spatial_size': (16, 16),
                             'hist_bins': 16, 'spatial_feat': True,
                             'hist_feat': True, 'hog_feat': True, 'probability': True}
        self.heat = None
        self.thres_cnt = 0
        self.cnt = 0
        self.DEBUG = True
        self.frame_nr = 0
        self.use_sliding_window = True
        self.parallel = 'serial' #'process'

    def fit(self, laneFit):
        model = Model()
        model_map = model.fit(self.model_config)

        self.laneFit = laneFit
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
        if True:
            # triangular shape of the ROI
            self.windows.append(slide_window_triangle(x_start_stop=(800, 1280), y_start_stop=(390, 550), 
                                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)))
            self.windows.append(slide_window_triangle(x_start_stop=(800, 1280), y_start_stop=(390, 550), 
                                        xy_window=(96, 96), xy_overlap=(0.5, 0.5)))
        else:
            self.windows.append(slide_window(x_start_stop=(800, 1280), y_start_stop=(390, 518), 
                                       xy_window=(64, 64), xy_overlap=(0.5, 0.5)))
            self.windows.append(slide_window(x_start_stop=(800, 1280), y_start_stop=(390, 518),
                                       xy_window=(96, 96), xy_overlap=(0.5, 0.5)))
             

    def process_image(self, img):
        # save original image in RGB for drawing results        
        img_orig = img.copy()

        if self.DEBUG:
            folder = 'debug/project_video/'
            #folder = 'debug/test_video/'

        # Quick merge of lane detecion
        if True:
            lane_image = self.laneFit.process_image(img_orig)
            img = lane_image.copy()
        
        draw_image = np.copy(img)

        # transform to selected colorspace of model
        img = cv2.cvtColor(img_orig, self.model_config['color_space'])

        # initialization of heat map
        if self.heat is None:
            self.heat = np.zeros_like(img[:,:,0]).astype(np.float)
        
        # processing on single windows or HOG feature extraction of complete image
        if self.use_sliding_window:
            if self.parallel == 'process':
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
            else:
                box_list = []
                for windows in self.windows:
                    box_list.extend(search_windows(img, windows, self.model_config, self.feat_extr))
        else:
            box_list = []
            sliding_window_desc = [(img, {'scale': 1.0, 'y_top': 400, 'y_bottom': 600, 'x_left': 640, 'x_right': 1280}),
                                   (img, {'scale': 1.5, 'y_top': 400, 'y_bottom': 650, 'x_left': 640, 'x_right': 1280}),
                                   (img, {'scale': 2.0, 'y_top': 400, 'y_bottom': 650, 'x_left': 640, 'x_right': 1280}),
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

        # heat of one image
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, box_list)
        heat = np.clip(heat, 0, 1)

        # Global heat map average
        alpha = 0.7
        self.heat = alpha * self.heat + heat
        self.heat = np.clip(self.heat, 0, 8)

        heat_thres = apply_threshold(self.heat.copy(), 3)
        
        heatmap_img = np.dstack((self.heat, np.zeros_like(heat), np.zeros_like(heat)))

        # Find final boxes from heatmap using label function
        labels = label(heat_thres)

        # draw the boxes
        window_img, boxes = draw_labeled_bboxes(draw_image, labels)

        # Try flood fill inside bounding boxes for better segmentation
        if False:
            ff = FloodFill()
            window_img = ff.fill(window_img, boxes)
        
        # debug: show single boxes and heatmap
        if False:
            window_img = draw_boxes(window_img, box_list, color=(0, 255, 0), thick=1)
            draw_image = cv2.addWeighted(draw_image, 1., 25*heatmap_img.astype(np.uint8), 1., 0)

        # show heatmap small at top right of image
        heatmap_small = cv2.resize(heatmap_img.astype(np.uint8), (0, 0), fx=0.15, fy=0.15).reshape((108, 192, 3)) * 255
        offset_small = 26
        offset_x_small = 600
        window_img[offset_small:offset_small + 108, offset_x_small+484:offset_x_small+676, :] = heatmap_small

        # save the single images for debugging
        if self.DEBUG:
            window_img_bgr = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder + 'processed/frame_{:04d}.jpg'.format(self.frame_nr), window_img_bgr)
            self.frame_nr += 1

        return window_img    


if __name__ == "__main__":
    def main():
        img = cv2.imread('test_images/test1.jpg')
        img = cv2.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        procimg = ProcessImage()
        procimg.fit(None)
        boxes = procimg.windows[0]+procimg.windows[1]
        print(boxes)
        img = draw_boxes(img, boxes, color=(0, 255, 0), thick=2)        
        img = cv2.image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output_images/boxes.jpg', img)

    main()
