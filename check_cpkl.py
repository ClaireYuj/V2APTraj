#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   check_cpkl.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/10/19 13:32
------------      --------    -----------

"""
import pickle

# with open("save_dir/1/test_trajectories.cpkl","rb") as file:
with open("savedata/taxi_1/train_batch_cache.cpkl","rb") as file:
    data = pickle.load(file)
data = str(data)

with open("savedata/taxi_1/train_batch_cache.txt","w") as f:
    f.write(data)
# print(data)