#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: off2binvox.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: 将 ModelNet10 数据集中.off文件转为binvox文件
'''

import os, sys
import glob
import argparse

sys.path.append('..')
from mapping import CLASSES

# DOWNLOAD MACOS: https://www.patrickmin.com/binvox/download.php?id=6

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../dataset/ModelNet40', help='path to ModelNet40 dataset')
args = parser.parse_args()

def main(
    data_root,
    ):
    for c in CLASSES:
        for split in ['test', 'train']:
            for off in glob.glob(os.path.join(data_root, c, split, '*.off')):
                binname = os.path.join(data_root, c, split, os.path.basename(off).split('.')[0] + '.binvox')
                if os.path.exists(binname):
                    print(binname, "exits, continue...")
                    continue
                os.system(f'./binvox -d 32 -cb -pb {off}')

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
