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

from sklearn.model_selection import train_test_split

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
def search_windows(img, windows, clf, scaler, color_space='RGB', 
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



class ProcessImage():
    def __init__(self):
        self.model_config = {'color_space': 'YCrCb', 'orient': 11,
                             'pix_per_cell': 16, 'cell_per_block': 2,
                             'hog_channel': 'ALL', 'spatial_size': (16, 16),
                             'hist_bins': 16, 'spatial_feat': True,
                             'hist_feat': True, 'hog_feat': True}
        self.color_space = self.model_config['color_space'] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = self.model_config['orient']  # HOG orientations
        self.pix_per_cell = self.model_config['pix_per_cell'] # HOG pixels per cell
        self.cell_per_block = self.model_config['cell_per_block'] # HOG cells per block
        self.hog_channel = self.model_config['hog_channel'] # Can be 0, 1, 2, or "ALL"
        self.spatial_size = self.model_config['spatial_size'] # Spatial binning dimensions
        self.hist_bins =self.model_config['hist_bins']    # Number of histogram bins
        self.spatial_feat = self.model_config['spatial_feat'] # Spatial features on or off
        self.hist_feat = self.model_config['hist_feat'] # Histogram features on or off
        self.hog_feat = self.model_config['hog_feat'] # HOG features on or off
        self.y_start_stop = [350, 650] # Min and max in y to search in slide_window()
        self.x_start_stop = [640, 1280]
        self.heat = None
        self.thres_cnt = 0
        self.cnt = 0
        self.DEBUG = True
        self.frame_nr = 0
        self.proba = True

    def fit_new_model(self, pickle_file):
        # Read in cars and notcars
        cars = []
        notcars = []

        self.frame_nr = 0

        carimages = glob.glob('dataset/vehicles/*/*.png')
        carimages.extend(glob.glob('dataset/vehicles/*/*.jpg'))
        print("Number of car samples {}".format(len(carimages)))
        for image in carimages:
            cars.append(image)

        noncarimages = glob.glob('dataset/non-vehicles/*/*.png')
        noncarimages.extend(glob.glob('dataset/non-vehicles/*/*.jpg'))
        print("Number of non-car samples {}".format(len(noncarimages)))
        for image in noncarimages:
            notcars.append(image)
                

        # Reduce the sample size because
        # The quiz evaluator times out after 13s of CPU time
        sample_size = None

        if sample_size is not None:
            cars = cars[0:sample_size]
            notcars = notcars[0:sample_size]

        car_features = extract_features(cars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        print('Car features extracted, len = {}  shape = {}'.format(len(car_features), car_features[0].shape))

        notcar_features = extract_features(notcars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        print('Car features extracted, len = {}  shape = {}'.format(len(notcar_features), notcar_features[0].shape))
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        print('X.shape = {}'.format(X.shape))

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler
        self.x_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.x_scaler.transform(X_train)
        X_test = self.x_scaler.transform(X_test)

        print('Using:',self.orient,'orientations',self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        t=time.time()

        parameters = [{'kernel':('linear',), 'C': list(np.arange(0.5, 3., 0.1))},
                    #{'kernel':('rbf',), 'C': list(np.arange(0.5, 10., 0.1)), 'gamma': [0.002, 0.001, 0.007, 0.0005]},
                    ]
        
        if False:
            print('Searching for best classifier...')
            svr = svm.SVC()
            clf = GridSearchCV(svr, parameters)
            clf.fit(X_train, y_train)

            print(clf.best_params_)
        else:
            print('Fitting train set...')
            clf = svm.SVC(kernel='linear', C=0.01, probability=self.proba)
            clf.fit(X_train, y_train)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

        self.clf = clf

        with pickle_file.open(mode='wb') as f:
            model_map = {'model': self.clf, 'x_scaler': self.x_scaler, 'model_config': self.model_config}
            pickle.dump(model_map, f)

    def fit(self):
        pickle_file = Path('SVM.p')

        if pickle_file.is_file():
            with pickle_file.open(mode=('rb')) as f:
                print('Loading model {}'.format(pickle_file.name))
                try:
                    model_map = pickle.load(f)
                    self.clf = model_map['model']
                    self.x_scaler = model_map['x_scaler']

                    if 'model_config' in model_map:
                        print('Loading model parameter from file {}'.format(pickle_file.name))
                        self.model_config = model_map['model_config']
                        self.color_space = self.model_config['color_space']
                        self.orient = self.model_config['orient']
                        self.pix_per_cell = self.model_config['pix_per_cell']
                        self.cell_per_block = self.model_config['cell_per_block']
                        self.hog_channel = self.model_config['hog_channel']
                        self.spatial_size = self.model_config['spatial_size']
                        self.hist_bins =self.model_config['hist_bins']
                        self.spatial_feat = self.model_config['spatial_feat']
                        self.hist_feat = self.model_config['hist_feat']
                        self.hog_feat = self.model_config['hog_feat']
                    else:
                        print('\033[93mWarning: model parameter not contained in {}\033[0m'.format(pickle_file.name))
                except EOFError:
                    print('\033[93mWarning: error in {} - fit model from scratch\033[0m'.format(pickle_file.name))
                    self.fit_new_model(pickle_file)
        else:
            self.fit_new_model(pickle_file)

        self.find_cars = FindCars()

        swd = {'clf': self.clf, 'x_scaler': self.x_scaler, 'use_spatial': self.spatial_feat,
                'use_color': self.hist_feat,
                'use_hog': self.hog_feat, 'proba': self.proba, 'pix_per_cell': self.pix_per_cell,
                'cell_per_block': self.cell_per_block, 'spatial_size': self.spatial_size,
                'hist_bins': self.hist_bins, 'orient': self.orient}

        self.find_cars.fit(swd)


    def process_image(self, img):
        if self.DEBUG:
            folder = 'debug/project_video/'
            #folder = 'debug/test_video/'
            if self.frame_nr < 0:
                self.frame_nr += 1
                return img
        #plt.imshow(img)
        #plt.show()
        #bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('test.jpg', bgr)
        if self.heat is None:
            self.heat = np.zeros_like(img[:,:,0]).astype(np.uint8)

        draw_image = np.copy(img)
        
        if False:
            windows = slide_window(img, x_start_stop=[None, None], y_start_stop=self.y_start_stop, 
                                xy_window=(96, 96), xy_overlap=(0.5, 0.5))

            box_list = search_windows(img, windows, self.clf, self.x_scaler, color_space=self.color_space, 
                                    spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                    orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                    cell_per_block=self.cell_per_block, 
                                    hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                    hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        else:
            box_list = []
            sliding_window_desc = [#{'scale': 0.5, 'y_top': 400, 'y_bottom': 500, 'x_left': 640, 'x_right': 1280},
                                   #{'scale': 0.7, 'y_top': 400, 'y_bottom': 550, 'x_left': 640, 'x_right': 1280},
                                   {'scale': 1.0, 'y_top': 400, 'y_bottom': 600, 'x_left': 640, 'x_right': 1280},
                                   {'scale': 1.5, 'y_top': 400, 'y_bottom': 650, 'x_left': 640, 'x_right': 1280}
                                    ]
            for swd in sliding_window_desc:
                box, himg = self.find_cars.find_cars(img, swd)
                #if scale == 1.0:
                #    hog_img = himg
                
                box_list.extend(box)

        self.heat = add_heat(self.heat, box_list)

        self.heat = apply_threshold(self.heat, 1)
        
        self.heat = np.clip(self.heat, 0, 7)

        heatmap_img = np.dstack((self.heat, np.zeros_like(self.heat), np.zeros_like(self.heat)))

        draw_image = cv2.addWeighted(draw_image, 1., 25*heatmap_img, 1., 0)

        #hog_image = np.zeros_like(draw_image)
        #hog_image[self.y_start_stop[0]:self.y_start_stop[1], self.x_start_stop[0]:self.x_start_stop[1], 2] = hog_img
        #hog_image[self.y_start_stop[0]:self.y_start_stop[1], self.x_start_stop[0]:self.x_start_stop[1], 1] = hog_img
        #hog_img = np.dstack((np.zeros_like(hog_img), np.zeros_like(hog_img), hog_img))
        #hog_img = cv2.resize(hog_img, (draw_image.shape[1], draw_image.shape[0]))
        #draw_image = cv2.addWeighted(draw_image, 0.1, hog_image, 1., 0)
        

        # Find final boxes from heatmap using label function
        labels = label(self.heat)

        window_img = draw_labeled_bboxes(draw_image, labels)
        
        window_img = draw_boxes(window_img, box_list, color=(0, 255, 0), thick=1)
        #window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        if self.DEBUG:
            cv2.imwrite(folder + 'original/frame_{:04d}.jpg'.format(self.frame_nr), img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            window_img_bgr = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder + 'original/frame_{:04d}.jpg'.format(self.frame_nr), img_bgr)
            cv2.imwrite(folder + 'processed/frame_{:04d}.jpg'.format(self.frame_nr), window_img_bgr)
            self.frame_nr += 1

        return window_img    



