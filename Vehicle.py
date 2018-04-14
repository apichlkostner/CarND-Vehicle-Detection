import numpy as np
import cv2
import time
from HelperFunctions import *
from FeatureExtract import FeatureExtractor

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

        self.feat_extr = FeatureExtractor(self.model_config) 

    def fit_boxes(self, boxes):
        x = 5
        #for box in boxes:
        #    if box[0][0] >

    def init(self, pos, velocity, bbox, feat_extr=None):
        self.pos = pos
        self.velocity = velocity
        self.bbox = bbox

        if feat_extr is not None:
            self.feat_extr = feat_extr

    def update(self, img):
        self.pos += self.velocity

        p1 = self.bbox[0]
        p2 = self.bbox[1]

        features = []
        cnt = 0

        for y in range(p1[0], p2[0], 16):
            for x in range(p1[1], p2[1], 16):
                feature_image = img[y:y+64, x:x+64]

                cnt += 1

                features = self.feat_extr.calc_features(feature_image)

                

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