import csv
import cv2
import numpy as np
import pandas as pd
import sys
from moviepy.editor import VideoFileClip
from DataGeneration import *

def extract():
    basedir = 'dataset/object-dataset/'
    columns = ['image', 'x_min', 'y_min', 'x_max', 'y_max', 'x', 'label', 'color']
    df = pd.read_csv(basedir+'labels.csv', sep=' ', names=columns, header=None)
    
    print('Unique labels: ' + str(df['label'].unique()))

    for index, row in df.iterrows():
        print('Reading row nr {} ({})'.format(index, row['image']))
        img_bgr = cv2.imread(basedir + row['image'])
        img_small = cv2.resize(img_bgr[row['y_min']:row['y_max'], row['x_min']:row['x_max']], (64, 64))
        if (row['label'] == 'car') or (row['label'] == 'truck'):
            cv2.imwrite('dataset/vehicles/udacity/{}.jpg'.format(index), img_small)
        else:
            cv2.imwrite('dataset/non-vehicles/udacity/{}.jpg'.format(index), img_small)

        

def main():
    if (len(sys.argv) > 1) and isinstance(sys.argv[1], str):
        filename = sys.argv[1]
    else:
        filename = 'test_video.mp4'
    
    print('Processing file ' + filename)

    clip1 = VideoFileClip('source_videos/' + filename)#.subclip(0,5)

    gen = TrainingDataGenerator()

    for frame in clip1.iter_frames():
        gen.create_training_data(frame)
    

if __name__ == "__main__":
    main()