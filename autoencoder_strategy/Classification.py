"""
In this file we want to use the encoded state to train a classifier on the task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append('../')
from dataset.voxelDataset import VoxelDataset
from dataset.PointCloudDataset import PointCloudDataset
from voxel_ae import voxAutoEncoder
from FoldingNet import FoldNet

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_scores (target, predictions):
    results = {}
    results['accuracy'] = accuracy_score(target, predictions)
    results['recall'] = recall_score(target, predictions, average='macro')
    results['precision'] = precision_score(target, predictions, average='macro')
    results['f1'] = f1_score(target, predictions, average='macro')
    # dataframe
    df = pd.DataFrame(results , index=[0])
    return df

def create_encoded_states(model, dataset):
    """
    Given a model and a dataset, it returns the encoded states of the dataset
    """
    model.eval()
    encoded_states = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataset):
            voxel, label = data
            #unsqueeze to add batch dimension
            voxel = voxel.unsqueeze(0)
            voxel = voxel.to(DEVICE)
            encoded = model.encode(voxel)
            encoded_states.append(encoded.cpu().numpy().flatten())
            labels.append(label)
    return encoded_states, labels


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints_path', type=str, default='../checkpoints')
parser.add_argument('--model_name', type=str, default='vox')
parser.add_argument('--data_dir', type=str, default='../dataset')
parser.add_argument('--create_dataset', type=bool, default=False)
parser.add_argument('--ndata', type=int, default=-1, help='Number of data ')
parser.add_argument('--train', type=bool, default=True, help='Train or test')

def main (
    checkpoints_path: str = '../checkpoints',
    model_name: str = 'vox',
    data_dir: str = '../dataset/svm_dataset',
    create_dataset: bool = False,
    ndata: int = -1,
    train: bool = True,
    ):
    import numpy as np

    # data_dir
    data_dir = os.path.join(data_dir, model_name)

    checkpoints_path  = os.path.join(checkpoints_path, model_name)
    if create_dataset:
        if model_name == 'vox':
            # create dataset of encoded states
            input_shape = (32, 32, 32)
            dataset_train  = VoxelDataset('../dataset/ModelNet40', 
                                            train=True, 
                                        )
            dataset_val    = VoxelDataset('../dataset/ModelNet40', 
                                            train=False, 
                                            )

            # get subset of dataset random samples
            if ndata > 0:
                np.random.seed(0)
                train_indices = np.random.choice(len(dataset_train), ndata, replace=False)
                val_indices = np.random.choice(len(dataset_val), ndata, replace=False)

                dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
                dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

            print (f"Train dataset size: {len(dataset_train)}")
            print (f"Val dataset size: {len(dataset_val)}")

            model = voxAutoEncoder(input_shape).to(DEVICE)
        
        elif model_name == 'foldingnet':
            dataset_train  = PointCloudDataset('../dataset/modelnet40_normal_resampled', 
                                            train=True, 
                                            ndata=4000, 
                                            file_extension='.txt', 
                                            npoints=4000
                                        )
            if train: test_data =2000
            else: test_data = -1
            npoints = 4000
            dataset_val    = PointCloudDataset('../dataset/modelnet40_normal_resampled', 
                                                train=False, 
                                                ndata=test_data,
                                                file_extension='.txt', 
                                                npoints=npoints
                                            )

            print (f"Train dataset size: {len(dataset_train)}")
            print (f"Val dataset size: {len(dataset_val)}")

            # get subset of dataset random samples
            if ndata > 0:
                np.random.seed(0)
                train_indices = np.random.choice(len(dataset_train), ndata, replace=False)
                val_indices = np.random.choice(len(dataset_val), ndata, replace=False)

                dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
                dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

            print (f"Train dataset size: {len(dataset_train)}")
            print (f"Val dataset size: {len(dataset_val)}")

            model = FoldNet(npoints).to(DEVICE)


        # load model if exists
        models_saved = glob.glob(os.path.join(checkpoints_path, 'model_*.pth'))
        if len(models_saved) > 0:
            # get most recent model
            epoches_done = max([int(model.split('_')[-1].split('.')[0]) for model in models_saved])
            model_path = os.path.join(checkpoints_path, f'model_{epoches_done}.pth')
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))

        # # plot an example of reconstruction
        # sample = dataset_val[100]
        # voxel, label = sample
        # voxel = voxel.unsqueeze(0)
        # voxel = voxel.to(DEVICE)
        # decoded, encoded = model(voxel)
        # decoded = decoded.detach().cpu().numpy().squeeze()
        # voxel = voxel.cpu().numpy().squeeze()

        # # plot
        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # ax.voxels(voxel, edgecolor='k')
        # ax.set_title('Original')
        # ax = fig.add_subplot(122, projection='3d')
        # ax.voxels(decoded, edgecolor='k')
        # ax.set_title('Reconstructed')
        # plt.show()

        encoded_states_train, labels_train = create_encoded_states(model, dataset_train)
        encoded_states_val, labels_val = create_encoded_states(model, dataset_val)


        # save encoded states and labels
        print ('='*20, 'SAVING ENCODED STATES', '='*20)

        # create dir if not exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        np.save(os.path.join(data_dir, 'encoded_states_train.npy'), encoded_states_train)
        np.save(os.path.join(data_dir, 'encoded_states_val.npy'), encoded_states_val)
        np.save(os.path.join(data_dir, 'labels_train.npy'), labels_train)
        np.save(os.path.join(data_dir, 'labels_val.npy'), labels_val)

    else:
        # load encoded states and labels
        print ('='*20, 'LOADING ENCODED STATES', '='*20)
        import numpy as np
        encoded_states_train = np.load(os.path.join(data_dir, 'encoded_states_train.npy'))
        encoded_states_val = np.load(os.path.join(data_dir, 'encoded_states_val.npy'))
        labels_train = np.load(os.path.join(data_dir, 'labels_train.npy'))
        labels_val = np.load(os.path.join(data_dir, 'labels_val.npy'))

    print (f"Train dataset size: {len(encoded_states_train)}")
    print (f"Val dataset size: {len(encoded_states_val)}")


    #############################
    # CLASSIFICATION
    #############################

    ############# SVM #############
    SVM = True
    if SVM:
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier

        print ('='*20, 'TRAINING SVM CLASSIFIER', '='*20)

        # if best params file exists, load it
        best_params_file = os.path.join('params', model_name+'_SVMbest_params.txt')
        if os.path.exists(best_params_file):
            best_params = {}
            with open(best_params_file, 'r') as f:
                for line in f:
                    line = line.replace('\'', '').replace('{', '').replace('}', '').replace(' ', '') # remove spaces and brackets
                    line = line.split(',')
                    for param in line:
                        param = param.split(':')
                        best_params[param[0]] = float(param[1]) if param[1].replace('.', '').isnumeric() else param[1]
            print ('best params', best_params)
            clf = SVC(**best_params)
        else:
            print ("doing grid search")
            # grid search
            param_grid = {'C': [0.1,1, 10, 100], 
                        'gamma': [1,0.1,0.01,0.001],
                        'kernel': ['rbf', 'poly', 'sigmoid']}

            from sklearn.model_selection import GridSearchCV
            grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv=5)
            grid.fit(encoded_states_train, labels_train)
            print('best params', grid.best_params_)
            print('best estimator', grid.best_estimator_)

            # save best params
            if not os.path.exists('params'):
                os.makedirs('params')
            with open(os.path.join('params', 'SVMbest_params.txt'), 'w') as f:
                f.write(str(grid.best_params_))

            clf = grid.best_estimator_

        clf.fit(encoded_states_train, labels_train)
        predictions = clf.predict(encoded_states_train)
        train_scores = get_scores(labels_train, predictions)
        # save 
        print (f"Train scores: \n{train_scores}")

        predictions = clf.predict(encoded_states_val)
        train_scores = get_scores(labels_val, predictions)
        print (f"test scores: \n{train_scores}")

    ############# FFNN #############
    print ('='*20, 'TRAINING FFNN CLASSIFIER', '='*20)
    from FFClassifier import classifier

    # create a dataset/loader for the encoded states
    # convert to numpy
    encoded_states_train = np.array(encoded_states_train)
    encoded_states_val = np.array(encoded_states_val)
    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)

    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(encoded_states_train).float(), torch.from_numpy(labels_train).long())
    dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(encoded_states_val).float(), torch.from_numpy(labels_val).long())

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=False)

    FFclf = classifier(encoded_states_train.shape[1], len(np.unique(labels_train)))

    if os.path.exists(os.path.join(FFclf.save_dir, 'best.pth')):
        print ('Loading FFNN classifier from {}'.format(os.path.join(FFclf.save_dir, 'best.pth')))
        FFclf.load('best.pth')
    else:
        FFclf.fit(train_loader, val_loader, epochs=300)
        # load best model
        FFclf.load('best.pth')

    print ('='*20, 'EVALUATING FFNN CLASSIFIER', '='*20)
    train_predictions = FFclf.predict(train_loader)
    train_scores = get_scores(labels_train, train_predictions)

    test_predictions = FFclf.predict(val_loader)
    test_scores = get_scores(labels_val, test_predictions)

    # save
    print (f"Train scores: \n{train_scores}")
    print (f"Test scores: \n{test_scores}")



        

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))