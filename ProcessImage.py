import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
import sklearn.svm as svm
import sklearn.grid_search as grid_search
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from HelperFunctions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

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
        self.color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 11  # HOG orientations
        self.pix_per_cell = 16 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16) # Spatial binning dimensions
        self.hist_bins = 16    # Number of histogram bins
        self.spatial_feat = False # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.y_start_stop = [350, 720] # Min and max in y to search in slide_window()

    def fit(self):
        # Read in cars and notcars
        cars = []
        notcars = []

        carimages = glob.glob('dataset/vehicles/*/*.png')
        print("Number of car samples {}".format(len(carimages)))
        for image in carimages:
            cars.append(image)

        noncarimages = glob.glob('dataset/non-vehicles/*/*.png')
        print("Number of non-car samples {}".format(len(noncarimages)))
        for image in noncarimages:
            notcars.append(image)
                

        # Reduce the sample size because
        # The quiz evaluator times out after 13s of CPU time
        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

        car_features = extract_features(cars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = extract_features(notcars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        print('Using:',self.orient,'orientations',self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        t=time.time()

        parameters = [{'kernel':('linear',), 'C': (1.,)}, #list(np.arange(1., 1., 1.))},
                    #{'kernel':('rbf',), 'C': list(np.arange(1, 25., 0.5)), 'gamma': [0.001, 0.007, 0.0005, 0.0002, 0.0001]},
                    ]
        svr = svm.SVC()
        clf = grid_search.GridSearchCV(svr, parameters)
        clf.fit(X_train, y_train)

        print(clf.best_params_)

        # Use a linear SVC 
        #svc = LinearSVC()
        # Check the training time for the SVC

        #svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()

        self.clf = clf


    def process_image(self, img):
        #plt.imshow(img)
        #plt.show()
        #bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('test.jpg', bgr)
        draw_image = np.copy(img)
        print('Image min={} max={}'.format(img.min(), img.max()))
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        img = img.astype(np.float32) / 255

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=self.y_start_stop, 
                            xy_window=(96, 96), xy_overlap=(0.5, 0.5))

        hot_windows = search_windows(img, windows, self.clf, self.X_scaler, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)                       

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        return window_img    

#plt.imshow(window_img)
#plt.show()


