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
with open("taxiData/1/train_trajectories.cpkl","rb") as file:
    data = pickle.load(file)
data = str(data)

with open("taxiData/1/train_trajectories.txt","w") as f:
    f.write(data)
# print(data)