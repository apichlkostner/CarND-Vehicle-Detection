import numpy as np
import cv2
import time
from HelperFunctions import *

class Vehicle():
    def __init__(self):
        self.pos = np.array([0, 0]).astype(np.float)
        self.velocity = np.array([0, 0]).astype(np.float)
        self.bbox = np.array([[0, 0], [0, 0]]).astype(np.int)

        self.model_config = {'color_space': 'YCrCb', 'orient': 9,
                             'pix_per_cell': 16, 'cell_per_block': 4,
                             'hog_channel': 'ALL', 'spatial_size': (16, 16),
                             'hist_bins': 16, 'spatial_feat': True,
                             'hist_feat': True, 'hog_feat': True, 'probability': True}    

    def fit_boxes(self, boxes):
        x = 5
        #for box in boxes:
        #    if box[0][0] >

    def init(self, pos, velocity, bbox):
        self.pos = pos
        self.velocity = velocity
        self.bbox = bbox

    def update(self, img):
        self.pos += self.velocity

        spatial_feat = True
        hist_feat = True
        hog_feat = True

        p1 = self.bbox[0]
        p2 = self.bbox[1]

        file_features = []
        cnt = 0

        for y in range(p1[0], p2[0], 16):
            for x in range(p1[1], p2[1], 16):
                feature_image = img[y:y+64, x:x+64]

                cnt += 1

                # spatial features
                if spatial_feat:
                    spatial_features = bin_spatial(feature_image, size=self.model_config['spatial_size'])
                    file_features.append(spatial_features)

                # color histogram features
                if hist_feat:
                    # Apply color_hist()
                    hist_features = color_hist(feature_image, nbins=self.model_config['hist_bins'])
                    file_features.append(hist_features)

                # hog features
                if hog_feat:
                # Call get_hog_features() with vis=False, feature_vec=True
                    if self.model_config['hog_channel'] == 'ALL':
                        hog_features = []
                        for channel in range(feature_image.shape[2]):
                            #hog_features.append(get_hog_features(feature_image[:,:,channel], 
                            #                    self.model_config['orient'], self.model_config['pix_per_cell']
                            #                    , self.model_config['cell_per_block'], vis=False, feature_vec=True))
                            winSize = (feature_image.shape[0], feature_image.shape[1])
                            blockSize = (winSize[0] // self.model_config['cell_per_block'],
                                         winSize[1] // self.model_config['cell_per_block'])
                            blockStride = (blockSize[0] // 2, blockSize[1] // 2)
                            cellSize = (self.model_config['pix_per_cell'], self.model_config['pix_per_cell'])
                            nbins = self.model_config['orient']
                            derivAperture = 1
                            winSigma = -1.
                            histogramNormType = 0
                            L2HysThreshold = 0.2
                            gammaCorrection = 1
                            nlevels = 64
                            useSignedGradients = True
                            #hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                            #            nbins, derivAperture, winSigma, histogramNormType, 
                            #            L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
                            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                                        nbins, derivAperture, winSigma, histogramNormType, 
                                        L2HysThreshold, gammaCorrection, nlevels)
                            descriptor = hog.compute(feature_image)
                            #print('Descriptor {}'.format(descriptor.shape))
                            hog_features.append(descriptor)

                        hog_features = np.ravel(hog_features)        
                    else:
                        hog_features = get_hog_features(feature_image[:,:,self.model_config['hog_channel']],
                                    self.model_config['orient'], self.model_config['pix_per_cell'],
                                    self.model_config['cell_per_block'], vis=False, feature_vec=True)
                    # Append the new feature vector to the features list
                    file_features.append(hog_features)

        print("Number of iterations = {}".format(cnt))
        #print('Feature vector shape = {}'.format(file_features))

def main():
    img = cv2.imread('test_images/test1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pos = np.array([888, 447]).astype(np.float)
    velocity = np.array([-2, -2]).astype(np.float)
    bbox = np.array([[414, 811], [493, 943]]).astype(np.int)

    t0 = time.time()

    vehicle = Vehicle()
    vehicle.init(pos, velocity, bbox)
    vehicle.update(img)


    t1 = time.time()

    print('Time = {}s'.format(t1-t0))

if __name__ == "__main__":
    main()