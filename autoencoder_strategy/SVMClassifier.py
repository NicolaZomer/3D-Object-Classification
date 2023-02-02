"""
In this file we want to use the encoded state to train a classifier on the task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys, os
import glob
from tqdm import tqdm


sys.path.append('../')
from dataset.voxelDataset import VoxelDataset
from voxel_ae import voxAutoEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            encoded_states.append(encoded)
            labels.append(label)
    model.train()
    return encoded_states, labels


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints_path', type=str, default='../checkpoints')
parser.add_argument('--model', type=str, default='vox')
parser.add_argument('--data_dir', type=str, default='../dataset/svm_dataset')
parser.add_argument('--create_dataset', type=bool, default=True)

def main (
    checkpoints_path: str = '../checkpoints',
    model: str = 'vox',
    data_dir: str = '../dataset/svm_dataset',
    create_dataset: bool = False,
    ):


    checkpoints_path  = os.path.join(checkpoints_path, 'vox')

    if create_dataset:
        # create dataset of encoded states
        input_shape = (32, 32, 32)
        dataset_train  = VoxelDataset('../dataset/ModelNet40', 
                                        train=True, 
                                    )
        dataset_val    = VoxelDataset('../dataset/ModelNet40', 
                                        train=False, 
                                        )


        print (f"Train dataset size: {len(dataset_train)}")
        print (f"Val dataset size: {len(dataset_val)}")

        model = voxAutoEncoder(input_shape).to(DEVICE)

        # load model if exists
        models_saved = glob.glob(os.path.join(checkpoints_path, 'model_*.pth'))
        if len(models_saved) > 0:
            # get most recent model
            epoches_done = max([int(model.split('_')[-1].split('.')[0]) for model in models_saved])
            model_path = os.path.join(checkpoints_path, f'model_{epoches_done}.pth')
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))

        encoded_states_train, labels_train = create_encoded_states(model, dataset_train)
        encoded_states_val, labels_val = create_encoded_states(model, dataset_val)


        # save encoded states and labels
        print ('='*20, 'SAVING ENCODED STATES', '='*20)
        # to numpy
        encoded_states_train = [encoded.cpu().numpy() for encoded in encoded_states_train]
        encoded_states_val = [encoded.cpu().numpy() for encoded in encoded_states_val]
        labels_train = [label.cpu().numpy() for label in labels_train]
        labels_val = [label.cpu().numpy() for label in labels_val]

        # save
        import numpy as np

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


    # train SVM classifier
    print ('='*20, 'TRAINING SVM CLASSIFIER', '='*20)
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    clf = SVC(gamma='auto')
    clf.fit(encoded_states_train, labels_train)

    print (f"Train accuracy: {accuracy_score(labels_train, clf.predict(encoded_states_train))}")
    print (f"Val accuracy: {accuracy_score(labels_val, clf.predict(encoded_states_val))}")

        

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))