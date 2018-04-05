import numpy as np
import cv2
import glob
import time
from pathlib import Path
from sklearn.svm import LinearSVC, SVC
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from skimage import data, exposure
import pickle
#from skimage.feature import hog
from HelperFunctions import *
from FindCars import *
from Segmentation import *

from sklearn.model_selection import train_test_split
import concurrent.futures

def fit_model(args):
    """
    Fits model to train set and checks agains test set
    It's not included to a class since it should be called as a separate process.
    Inside a class the complete class would be pickled and copied to the
    address space of the new process which needs much ram.
    """
    print('Fitting model with C={}'.format(args['C']))
    t0 = time.time()
    clf = svm.SVC(kernel='linear', C=args['C'], probability=args['proba'])
    clf.fit(args['x_train'], args['y_train'])
    test_score = clf.score(args['x_test'], args['y_test'])
    t1 = time.time()
    print('Fitted model with C={} in time {:.1f}'.format(args['C'], t1 - t0))

    return clf, test_score

class Model():
    def __init__(self):
        self.x = 0

    def fit_new_model(self, pickle_file, model_config):
        main_path = 'dataset/split/'
        carimages = glob.glob(main_path + 'train/car/*/*.png')
        carimages.extend(glob.glob(main_path + 'train/car/*/*.jpg'))
        print("Number of car training samples {}".format(len(carimages)))

        noncarimages = glob.glob(main_path + 'train/noncar/*/*.png')
        noncarimages.extend(glob.glob(main_path + 'train/noncar/*/*.jpg'))
        print("Number of non-car training samples {}".format(len(noncarimages)))                

        train_data = {'car': carimages, 'noncar': noncarimages}

        main_path = 'dataset/split/'
        carimages = glob.glob(main_path + 'test/car/*/*.png')
        carimages.extend(glob.glob(main_path + 'test/car/*/*.jpg'))
        print("Number of car test samples {}".format(len(carimages)))

        noncarimages = glob.glob(main_path + 'test/noncar/*/*.png')
        noncarimages.extend(glob.glob(main_path + 'test/noncar/*/*.jpg'))
        print("Number of non-car test samples {}".format(len(noncarimages)))                

        test_data = {'car': carimages, 'noncar': noncarimages}
        
        # for faster test sample size can be reduced
        sample_size = None

        if sample_size is not None:
            train_data['car'] = train_data['car'][0:sample_size]
            train_data['noncar'] = train_data['noncar'][0:sample_size]
            test_data['car'] = test_data['car'][0:sample_size // 5]
            test_data['noncar'] = test_data['noncar'][0:sample_size // 5]
            

        t0 = time.time()

        arguments = [(train_data['car'], model_config),
                     (train_data['noncar'], model_config),
                     (test_data['car'], model_config),
                     (test_data['noncar'], model_config)]
        input_type = ['train_cars', 'train_noncars', 'test_cars', 'test_noncars']

        data = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for category, feature_data in zip(input_type, executor.map(extract_features_map, arguments)):
                data[category] = feature_data
                print('{} features extracted, len = {}  shape = {}'.format(category, len(feature_data),
                                                                         feature_data[0].shape))

        t1 = time.time()
        print('Feature extraction, time {:.2f}'.format(t1 - t0))
        
        # Create an array stack of feature vectors
        x_train = np.vstack((data['train_cars'], data['train_noncars'])).astype(np.float64)
        x_test  = np.vstack((data['test_cars'], data['test_noncars'])).astype(np.float64)
        print('x_train.shape = {}'.format(x_train.shape))
        print('x_test.shape = {}'.format(x_test.shape))

        # Define the labels vector
        y_train = np.hstack((np.ones(len(data['train_cars'])), np.zeros(len(data['train_noncars']))))
        y_test = np.hstack((np.ones(len(data['test_cars'])), np.zeros(len(data['test_noncars']))))
            
        # Fit a per-column scaler
        x_scaler = StandardScaler().fit(x_train)
        # Apply the scaler to X
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)

        print('Using: {} orientations {} pixels per cell and {} cells per block'.format(
            model_config['orient'], model_config['pix_per_cell'], model_config['cell_per_block']))
        print('Feature vector length:', len(x_train[0]))
        
        print('Searching for best parameters...')
        C_params = [0.01, 0.1, 1., 10.]
        models = {}
        score_max = 0.
        params = [{'C': C, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                   'proba': model_config['probability']}
                            for C in C_params]
        t2=time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for C, result in zip(C_params, executor.map(fit_model, params)):
                models[C] = result
                print('Model fitted with C={}, score={:.4f}'.format(C, result[1]))

                # save all model to be tested on video
                f_save = Path('saves/SVM_C{}_score{:.3f}.p'.format(C, result[1]))
                with f_save.open(mode='wb') as f:
                    model_map = {'model': result[0], 'x_scaler': x_scaler, 'model_config': model_config}
                    pickle.dump(model_map, f)

                # save best model
                if result[1] > score_max:
                    score_max = result[1]
                    clf = result[0]
        t3 = time.time()
        print('Time for searching parameter: {:.1f}'.format(t3 - t2))

        with pickle_file.open(mode='wb') as f:
            model_map = {'model': clf, 'x_scaler': x_scaler, 'model_config': model_config}
            pickle.dump(model_map, f)

            return model_map

    def fit(self, model_config):
        pickle_file = Path('SVM.p')

        if pickle_file.is_file():
            with pickle_file.open(mode=('rb')) as f:
                print('Loading model {}'.format(pickle_file.name))
                try:
                    model_map = pickle.load(f)

                    if 'model_config' in model_map:
                        print('Loading model parameter from file {}'.format(pickle_file.name))
                        model_config = model_map['model_config']
                    else:
                        print('\033[93mWarning: model parameter not contained in {}\033[0m'.format(pickle_file.name))
                except EOFError:
                    print('\033[93mWarning: error in {} - fit model from scratch\033[0m'.format(pickle_file.name))
                    model_map = self.fit_new_model(pickle_file, model_config)
        else:
            model_map = self.fit_new_model(pickle_file, model_config)

        return model_map