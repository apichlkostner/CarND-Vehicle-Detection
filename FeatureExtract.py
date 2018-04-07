import numpy as np
import cv2

class FeatureExtractor():
    def __init__(self, config):
        self.conf = config
    
    def calc_hog_features(self, img):
        hog_features = []

        for channel in range(img.shape[2]):
            winSize = (img.shape[0], img.shape[1])
            blockSize = (winSize[0] // self.conf['cell_per_block'],
                            winSize[1] // self.conf['cell_per_block'])
            blockStride = (blockSize[0] // 2, blockSize[1] // 2)
            cellSize = (self.conf['pix_per_cell'], self.conf['pix_per_cell'])
            nbins = self.conf['orient']
            derivAperture = 1
            winSigma = -1.
            histogramNormType = 0
            L2HysThreshold = 0.2
            gammaCorrection = 1
            nlevels = 64
            useSignedGradients = True

            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                        nbins, derivAperture, winSigma, histogramNormType, 
                        L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
            
            features = hog.compute(img)
            
            hog_features.append(features)

        hog_features = np.ravel(hog_features)

        return hog_features

    # Define a function to compute binned color features  
    def calc_spatial_features(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    # Define a function to compute color histogram features 
    def calc_color_features(self, img, nbins=32, bins_range=(0, 255)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        #print('Max {}  {}  {}'.format(channel1_hist[0],channel2_hist[0],channel3_hist[0]))
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def calc_features(self, img):
        features = []

        # spatial features
        if self.conf['spatial_feat']:
            spatial_features = self.calc_spatial_features(img, size=self.conf['spatial_size'])
            features.append(spatial_features)

        # color histogram features
        if self.conf['hist_feat']:
            # Apply color_hist()
            hist_features = self.calc_color_features(img, nbins=self.conf['hist_bins'])
            features.append(hist_features)

        # hog features
        if self.conf['hog_feat']:
            # Call get_hog_features() with vis=False, feature_vec=True
            hog_features = self.calc_hog_features(img)
            # Append the new feature vector to the features list
            features.append(hog_features)

        features = np.concatenate(features)

        return features

