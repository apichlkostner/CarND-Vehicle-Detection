import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

class FloodFill():
    def fill(self, img, boxes):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_gray = img[:,:,2]

        mask = np.zeros((img.shape[0]+2, img.shape[1]+2)).astype(np.uint8)

        kernel = np.ones((3,3),np.uint8)
        #kernel2 = np.ones((5,5),np.uint8)
        mask[1:-1, 1:-1] = cv2.Canny(img_gray,10,100)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask_org = mask.copy()

        for box in boxes:
            middle_point = ((box[0][0]+box[1][0])//2, (box[0][1]+box[1][1]-200)//2)
            middle_point = (1088, 471)
            #print("Box = {}, middlepoint = {}".format(box, middle_point))
            out = cv2.floodFill(img, mask, middle_point, (200, 200, 200), upDiff=(50,50,50), loDiff=(50,50,50))

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        rgbmask = np.dstack((mask_org, np.logical_xor(mask, mask_org), np.logical_xor(mask, mask_org)))
        rgbmask_small = cv2.resize(rgbmask.astype(np.uint8), (0, 0), fx=0.5, fy=0.5) * 200

        #(y,x,d) = rgbmask_small.shape
        #img[0:y, 0:x, :] = rgbmask_small

        return cv2.addWeighted(rgbmask[1:-1, 1:-1, :], 1, img, 0.5, 0)

def main():
    img = cv2.imread('test_images/test1.jpg')

    t0 = time.time()


    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print('Time conversion gray = {}'.format(time.time()-t0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = img[:,:,2]
    print('Time conversion = {}'.format(time.time()-t0))

    mask = np.zeros((img.shape[0]+2, img.shape[1]+2)).astype(np.uint8)

    kernel = np.ones((3,3),np.uint8)
    mask[1:-1, 1:-1] = cv2.Canny(img_gray,50,100)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 1)
    #cv2.Sobel(img[:,:,2],cv2.CV_64F,1,0,ksize=5)

    # out = cv2.floodFill(img, mask, (1140, 446), 200, upDiff=(6,5,5), loDiff=(6,5,5))
    # out = cv2.floodFill(img, mask, (898, 474), (200, 200, 200), upDiff=(20,20,30), loDiff=(20,20,30))
    out = cv2.floodFill(img, mask, (1140, 446), 200, upDiff=(10,10,10), loDiff=(10,10,10))
    out = cv2.floodFill(img, mask, (898, 474), (200, 200, 200), upDiff=(50,50,50), loDiff=(50,50,50))

    t1 = time.time()

    print('Time = {}s'.format(t1-t0))
    difval = 10
    #out = cv2.floodFill(img, mask, (898, 474), 200, upDiff=(difval,difval,difval,difval), loDiff=(difval,difval,difval,difval))
    #print(out[1])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt.imshow(img)
    #plt.imshow(mask, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
