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
from scipy.ndimage.measurements import label
from skimage import data, exposure
#from skimage.feature import hog
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

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block,
            spatial_size, hist_bins, color_space='RGB', frame_nr=0, folder=''):
    
    draw_img = np.copy(img)
    
    img_tosearch = img[ystart:ystop,xstart:,:]
    #img_tosearch = img_tosearch.astype(np.float32) / 255
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)      

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    t0 = time.time()
    #hog = np.zeros([3, ])

    if True and (scale == 1):
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        #ch1 = clahe.apply(ch1)
        hog1, img1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=True)
        hog2, img2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=True)
        hog3, img3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=True)
        #img1 = exposure.rescale_intensity(img1, in_range=(0, 10))
        #print('Shape = {} min = {}  max = {}'.format(img1.shape, img1.min(), img1.max()))
        #cv2.imwrite(folder + 'hog/frame_hog1_{:04d}.jpg'.format(frame_nr), img1)
        print('{} {} {}'.format(orient, pix_per_cell, cell_per_block))
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
        #cv2.imwrite(folder + 'hog/frame_hog2_{:04d}.jpg'.format(frame_nr), img2)
        #cv2.imwrite(folder + 'hog/frame_hog3_{:04d}.jpg'.format(frame_nr), img3)
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #print('Hog-shape = {}'.format(hog1.shape))
    t1 = time.time()

    #print('Time hog: {}'.format(t1 - t0))

    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
          
            # Get color features
            #spatial_features = np.array #bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            #spatial_features[np.isnan(spatial_features)] = 0.0
            #hist_features[np.isnan(hist_features)] = 0.0
            #hog_features[np.isnan(hog_features)] = 0.0
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))    #, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            
            proba = clf.predict_proba(test_features)
            #print(proba)
            test_prediction = 1 * (proba[0, 1] > 0.95)
            #print(test_prediction)
            test_prediction2 =clf.predict(test_features)
            #print('Prob = {}, Pred = {}'.format(proba, test_prediction2))
            #print(clf.predict_proba(test_features))
            
            if test_prediction == 1:
                xbox_left = np.int((xleft)*scale + xstart)
                ytop_draw = np.int((ytop)*scale + ystart)
                win_draw = np.int(window*scale)
                boxes.append([[xbox_left, ytop_draw], [xbox_left+win_draw, ytop_draw+win_draw]])
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
    
    t2 = time.time()

    #print('Time sliding window: {}'.format(t2 - t1))

    return boxes

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
        self.y_start_stop = [350, 650] # Min and max in y to search in slide_window()
        self.heat = None
        self.thres_cnt = 0
        self.cnt = 0
        self.DEBUG = True
        self.frame_nr = 0

    def fit(self):
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

        parameters = [{'kernel':('linear',), 'C': list(np.arange(0.5, 3., 0.1))},
                      #{'kernel':('rbf',), 'C': list(np.arange(0.5, 10., 0.1)), 'gamma': [0.002, 0.001, 0.007, 0.0005]},
                    ]
        
        if False:
            svr = svm.SVC()
            clf = grid_search.GridSearchCV(svr, parameters)
            clf.fit(X_train, y_train)

            print(clf.best_params_)
        else:
            #clf = svm.SVC(kernel='rbf', C=1.5, gamma=0.001)
            clf = svm.SVC(kernel='linear', C=1.5, probability=True)
            clf.fit(X_train, y_train)

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
        if self.DEBUG:
            #folder = 'debug/project_video/'
            folder = 'debug/test_video/'
        #plt.imshow(img)
        #plt.show()
        #bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('test.jpg', bgr)
        if self.heat is None:
            self.heat = np.zeros_like(img[:,:,0]).astype(np.uint8)

        draw_image = np.copy(img)
        
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        #img = img.astype(np.float32) / 255

        if False:
            windows = slide_window(img, x_start_stop=[None, None], y_start_stop=self.y_start_stop, 
                                xy_window=(96, 96), xy_overlap=(0.5, 0.5))

            box_list = search_windows(img, windows, self.clf, self.X_scaler, color_space=self.color_space, 
                                    spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                    orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                    cell_per_block=self.cell_per_block, 
                                    hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                    hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        else:
            box_list = []
            for scale in [1.0, 1.5, 2.0, 2.5, 3.0]:
                box_list.extend(find_cars(img, self.y_start_stop[0], self.y_start_stop[1], 640, scale, self.clf, self.X_scaler, self.orient,
                            self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins, self.color_space,
                            frame_nr=self.frame_nr, folder=folder))

        self.heat = add_heat(self.heat, box_list)

        self.heat = apply_threshold(self.heat, 2)

        #self.thres_cnt += 1
        #if self.thres_cnt > 10:
        #    self.heat = apply_threshold(self.heat, 5)
        #    self.thres_cnt = 0

        
        heatmap = np.clip(self.heat, 0, 10)

        heatmap_img = np.dstack((heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap)))

        #print('Type img {}    type heatimg {}'.format(type(img), type(heatmap_img)))

        draw_image = cv2.addWeighted(draw_image, 1., 25*heatmap_img, 1., 0)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

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

#plt.imshow(window_img)
#plt.show()


