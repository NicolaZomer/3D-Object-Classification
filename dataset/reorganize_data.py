# given a dataset, reorganize it into a new dataset with subfolder for train/test for each class

import os
import glob
import numpy as np

def reorganize_data(dataset_folder, train_ratio=0.8):
    
    # get subfolders
    classes = os.listdir(dataset_folder)
    for c in classes:
        if not os.path.isdir(os.path.join(dataset_folder, c)):
            classes.remove(c)
    classes.sort()

    for c in classes:
        # create train and test folders
        if not os.path.exists(os.path.join(dataset_folder, c, 'train')):
            os.makedirs(os.path.join(dataset_folder, c, 'train'))
        if not os.path.exists(os.path.join(dataset_folder, c, 'test')):
            os.makedirs(os.path.join(dataset_folder, c, 'test'))

        # move files
        all_files = glob.glob(os.path.join(dataset_folder, c, "*txt"))
        all_files.sort()


        # shuffle
        idx = np.arange(len(all_files))
        np.random.shuffle(idx)
        all_files = np.array(all_files)[idx]

        # split
        ntrain = int(len(all_files) * train_ratio)
        train_files = all_files[:ntrain]
        test_files = all_files[ntrain:]



        # move files
        for f in train_files:
            os.rename(f, os.path.join(dataset_folder, c, 'train', os.path.basename(f)))
        for f in test_files:
            os.rename(f, os.path.join(dataset_folder, c, 'test', os.path.basename(f)))


if __name__ == '__main__':
    reorganize_data('modelnet40_normal_resampled', train_ratio=0.8)