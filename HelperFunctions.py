import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from skimage.feature import hog
from cv2 import HOGDescriptor
from FeatureExtract import FeatureExtractor

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        USE_SKIMAGE = True
        if USE_SKIMAGE:
            features = hog(img, orientations=orient, 
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), 
                        block_norm= 'L2-Hys',
                        transform_sqrt=False, 
                        visualise=vis, feature_vector=feature_vec)
        else:
            features = HOGDescriptor((64,64), (16,16), 
            (pix_per_cell, pix_per_cell), (pix_per_cell, pix_per_cell), orient).compute(img)
        #print('Feature shape = {}'.format(features.shape))
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 255)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    #print('Max {}  {}  {}'.format(channel1_hist[0],channel2_hist[0],channel3_hist[0]))
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features_map(input):
    imgs, confmap = input    
    
    return extract_features(imgs, confmap)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, confmap):
    feat_extr = FeatureExtractor(confmap)

    # Create a list to append feature vectors to
    feature_vectors = []
    # Iterate through the list of images
    for file in imgs:
        #print('Processing {}'.format(file))

        # Read in each one by one
        image = cv2.imread(file)
        # Must be RGB like the images from the video
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape != (64, 64, 3):
            # ignore files with wrong shape
            print('Wrong image size {} of file {}'.format(image.shape, file))
            #assert(image.shape == (64, 64, 3))
        else:
            # apply color conversion
            color_space = confmap['color_space']
            feature_image = cv2.cvtColor(image, color_space)

            features = feat_extr.calc_features(feature_image)
            #print('Feature shape: {}'.format(features.shape))

            # append features of current file to list of feature vectors
            feature_vectors.append(features)

    # Return list of feature vectors
    return feature_vectors



def slide_window_triangle(x_start_stop, y_start_stop, xy_window, xy_overlap):
    x_stop = x_start_stop[1] - xy_window[0]
    y_stop = y_start_stop[1] - xy_window[1]
    delta_x = x_start_stop[1] - x_start_stop[0]
    delta_y = y_start_stop[1] - y_start_stop[0]
    alpha = (delta_x-380) / delta_y
    delta_step_x = int(xy_window[0] * (1 - xy_overlap[0]))
    delta_step_y = int(xy_window[1] * (1 - xy_overlap[1]))

    windows = []
    for y in range(y_start_stop[0], y_stop, delta_step_y):
        trix = int(alpha * (y - y_start_stop[0]))
        for x in range(x_start_stop[0] + trix, x_stop, delta_step_x):
            windows.append(((x, y), (x + xy_window[0], y + xy_window[1])))

    return windows

# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    #print('Heatmap reduce {}  max = {}'.format((heatmap >= 1).sum(), heatmap.max()))
    #num = heatmap[:, 1230:]
    #print('Right part min {}  max {}  mean {}'.format(num.min(), num.max(), num.mean()))
    #heatmap[heatmap >= 1] -= 1
    #print('Heatmap reduce after {}  max = {}'.format((heatmap >= 1).sum(), heatmap.max()))
    #print('Right part after min {}  max {}  mean {}'.format(num.min(), num.max(), num.mean()))

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

        boxes.append(bbox)
    # Return the image
    return img, boxes

def search_windows_slide(img, windows, feat_extr, x_scaler, model):
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
        #print('Feature 2 min={}  max={}'.format(features.min(), features.max()))
        test_features = x_scaler.transform(np.array(features).reshape(1, -1))

        if True:
            proba = model.predict_proba(test_features)
            prediction = 1 * (proba[0, 1] > 0.9)
        else:
            prediction = model.predict(test_features)

        if prediction == 1:
            on_windows.append(window)

    #8) Return windows for positive detections
    return on_windows

def find_cars_sliding(args):
    '''

    '''
    t0 = time.time()

    (img, windows, conf, feat_extr) = args
    x_scaler = conf['x_scaler']
    model = conf['model']

    box_list = search_windows_slide(img, windows, feat_extr, x_scaler, model)

    t1 = time.time()
    #print('Time for boxes: {:.3f}'.format(t1-t0))

    return box_list



if __name__ == "__main__":
    def main():
        a = slide_window_triangle((0, 100), (0, 100), (30, 30), (0.5, 0.5))
        print(a)
    main()