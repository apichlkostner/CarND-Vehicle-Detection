import csv
import cv2
import numpy as np
import pandas as pd
import sys
from moviepy.editor import VideoFileClip
from DataGeneration import *
import concurrent.futures

def extract_process(args):
    df = args['df']
    basedir = args['basedir']
    i = args['i']

    for index, row in df.iterrows():
        print('Process {} reading row nr {} ({})'.format(i, index, row['image']))
        img_bgr = cv2.imread(basedir + row['image'])
        img_small = cv2.resize(img_bgr[row['y_min']:row['y_max'], row['x_min']:row['x_max']], (64, 64))
        if (row['label'] == 'car') or (row['label'] == 'truck'):
            cv2.imwrite('dataset/vehicles/udacity/{}.jpg'.format(index), img_small)
        # else:
        #     cv2.imwrite('dataset/non-vehicles/udacity/{}.jpg'.format(index), img_small)

def extract():
    basedir = 'dataset/object-dataset/'
    columns = ['image', 'x_min', 'y_min', 'x_max', 'y_max', 'x', 'label', 'color']
    df = pd.read_csv(basedir+'labels.csv', sep=' ', names=columns, header=None)
    
    print('Unique labels: ' + str(df['label'].unique()))

    num_process = 4
    split_size = df.shape[0] // num_process
    process_arguments = [{'df': df[i*split_size:(i+1)*split_size], 'basedir': basedir, 'i': i}
                                 for i in range(num_process)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, exe in zip(range(num_process), executor.map(extract_process, process_arguments)):
            print('Finished process {}'.format(i))

    

        

def main():
    if False:
        if (len(sys.argv) > 1) and isinstance(sys.argv[1], str):
            filename = sys.argv[1]
        else:
            filename = 'test_video.mp4'
        
        print('Processing file ' + filename)

        clip1 = VideoFileClip('source_videos/' + filename)#.subclip(0,5)

        gen = TrainingDataGenerator()

        for frame in clip1.iter_frames():
            gen.create_training_data(frame)
    else:
        print('Extracting images from Udacity database...')
        extract()
    

if __name__ == "__main__":
    main()