import cv2
import numpy as np
import glob

class TrainingDataGenerator():
    def __init__(self):
        self.cnt = 0
        self.frame_nr = 0

    def create_training_data(self, img):
        print('Processing frame {}'.format(self.frame_nr))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.frame_nr += 1
        if self.frame_nr % 7 == 0:
            for x in range(32, 1280-96, 64):
                for y in range(400, 720-64, 64):
                    cv2.imwrite('dataset/non-vehicles/self/{}.jpg'.format(self.cnt), img[y:y+64, x:x+64])
                    self.cnt += 1
        return img

    def copy_training_data(self, files):
        imgs = []
        for f in files:
            imgs.append(cv2.imread(f))

        cnt = 0
        for i in range(1000):
            for img in imgs:
                cv2.imwrite('dataset/vehicles/video/bc{}.jpg'.format(cnt), img)
                cnt += 1

                
def main():
    tg = TrainingDataGenerator()
    if False:
        imgs = glob.glob('dataset/vehicles/video/*.jpg')
        tg.copy_training_data(imgs)
    else:
        tg.create_training_data()

if __name__ == "__main__":
    main()