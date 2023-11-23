#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_preprocessing.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/11/11 20:58
------------      --------    -----------

"""
import csv
import os

import pandas as pd
# input_file = "sample_taxi_data.csv"
from pyproj import Proj, Transformer, CRS

input_orderByCar_dir = "taxi/orderByCode/"
output_orderByCar_sample_file_path = "sample/22227_22228/sample_taxiData.csv"
input_file = "../data/TaxiData.txt"
output_file = "../data/true_pos_.csv"
sample_output_file = "taxi/taxi_id/true_pos_unsorted.csv"
# sample_output_file = "taxi/sample_taxiData.csv"
sample_sorted_output_file = "taxi/taxi_id/true_pos_.csv"
def get_data_in_frame(input_file=input_file,output_file=sample_output_file):
    """
    将数据按照第一行：时间（距离开始的时间差frame）
    第二行：id
    第三行：经度
    第四行：纬度
    """
    # global output_file
    # output_file = sample_output_file
    df = pd.DataFrame()
    time_list = []
    id_list = []
    lng_list = []
    lat_list = []
    print("start read: ",input_file)
    count = 0
    # with open(output_file, "w", newline="") as f:
    #     f.close()

    with open(input_file, "r") as read_f:

        for line in read_f.readlines():

            id = line.split(",")[0]
            time_str = line.split(",")[1]
            time = int(time_str.split(":")[0]) * 3600 + int(time_str.split(":")[1]) * 60 + int(time_str.split(":")[2])
            # print("time:",time, "time_str.split(\":\")[0]:",time_str.split(":")[0])
            lng = line.split(",")[2] # 经度
            lat = line.split(",")[3] # 纬度
            x, y = convert_latlon_to_xy(lng, lat)
            data = [time, id, x, y]
            # data = (range(len(data)), data)
            # time_list.append(time)
            # id_list.append(id)
            # lng_list.append(lng)
            # lat_list.append(lat)
            # data_in_col = zip(*data)
            # print(*data)
            count += 1
            df[count] = data

            print("-------finish write:", count, "-------------")

            time_list = []
            id_list = []
            lng_list = []
            lat_list = []


    with open(sample_output_file, "w") as write_f:
        df.to_csv(write_f,header=False, index=False)
    print("----------read and store--------------")



def get_sample_file(sample_rate = 0.1):
    count = 0
    sample_route = 1 / sample_rate
    global input_file
    print("start read:", input_file)
    with open(input_file, "r") as f_r:
        # with open("part_taxi_data.csv", "w", newline='') as f_w:
        with open("sample/22223_22224/sample_taxi_data.csv", "w", newline="") as f_w:
            for line in f_r.readlines():
                if count % sample_route == 0:
                    f_w.write(line)

                # else:
                #     print("---more than 30---")
                #     break
                count += 1
    print("-------ok-------")
    # input_file = "part_taxi_data.csv"
    input_file = "sample/22223_22224/sample_taxi_data.csv"

def time_sort_in_format(input_file=sample_output_file, output_file=sample_sorted_output_file,format=300):
    """
    把第一行时间按照每300s(5min)进行划分，并排序
    """
    df = pd.read_csv(input_file)
    print("try to sort the file:",input_file," by time")
    df = df.T
    df.index = pd.to_numeric(df.index).astype(int)  # 设置索引为浮点数型，直接设置为整型会报错
    for index, row in df.iterrows():

        new_index = int(index // format * format)
        df = df.rename(index={index: new_index})
    df.index = pd.to_numeric(df.index).astype(int)  # 设置索引为浮点数型，直接设置为整型会报错
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    # df.index = df.index.astype(float)
    df = df.sort_index(ascending=True) # sorted by time
    print("-------------------------------------------")
    id_name = df.columns[0]
    df[id_name] = df[id_name].astype(int)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    print(df.iloc[:, 0])
    df = df.T

    # print(df.ix[1])
    df.iloc[0, :] = df.iloc[0, :].astype(int)
    df.iloc[1, :] = df.iloc[1, :].astype(int)
    # df.ix[1] = df.ix[1].astype(int)


    with open(output_file, "w") as f:
        df.to_csv(f, index=False)



def id_sort_in_format(format=100):
    """
    把第一行时间按照每100 s进行划分，并按照id排序
    """
    df = pd.read_csv(sample_output_file)
    print("try to sort the file:",sample_output_file," by id")
    cols = []
    for column_label, column_data in df.iteritems():

        if column_label.split('.'):
            column_label = int(column_label.split('.')[0])
        new_col_label = int(int(column_label) // format * format)
        # df = df.rename(columns={column_label:new_col_label},inplace=True)
        cols.append(new_col_label)
    df.columns = cols


    with open(sample_sorted_output_file, "w") as f:
        df.to_csv(f, index=False)


def sort_in_col_by_time(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return sort_in_col_by_time(left) + middle +sort_in_col_by_time(right)

def convert_latlon_to_xy(longitude, latitude):
    """
    将经纬度转换为平面坐标，并只保留平面坐标的整数部分，且将整数部分转换为以千米为单位
    """

    # 转换经纬度为平面坐标
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395")
    x, y = transformer.transform(latitude, longitude)
    # x的整数部分
    x_integer_str = str(x).split(".")[0]
    len_x_integer = len(str(x).split(".")[0])
    # y的整数部分
    y_integer_str = str(y).split(".")[0]
    len_y_integer = len(str(y).split(".")[0])
    x = float(x_integer_str[:len_x_integer-3]+"."+ x_integer_str[-3:])
    y = float(y_integer_str[:len_y_integer-3]+"."+ y_integer_str[-3:])

    return x, y

def get_from_code2code(start_taxi_code=22523, end_taxi_code=22623, time_format=10, root_path="sample"): # 22223-34003 --  train set, 34004-36950:test set
    """
    从原始的taxiData中获得从start_taxiCode到end_taxi_code的文件，并拼成一个文件
    time_format: 把第一行时间按照每300 s进行划分
    """
    df = pd.DataFrame()
    root_path = "sample_lnglat/"
    print("start read: ",input_file)
    count = 0
    if not str(start_taxi_code)+ "_"+ str(end_taxi_code) in os.listdir(root_path):
        os.mkdir(root_path+ str(start_taxi_code) + "_"+ str(end_taxi_code))
    output_orderByCar_sample_file_path = root_path+ str(start_taxi_code)+ "_"+ str(end_taxi_code)+"/true_pos_.csv"
    # output_orderByCar_sample_file = open(output_orderByCar_sample_file_path, "w")
    for file in os.listdir(input_orderByCar_dir):
        if(file.endswith(".txt") and file.startswith("taxiCode_")):
            code = int(file.split("_")[1].split(".txt")[0])
            if code >= start_taxi_code and code <= end_taxi_code:
                with open(input_orderByCar_dir+file, "r") as read_f:
                    for line in read_f.readlines():
                        id = int(line.split(",")[0])
                        time_str = line.split(",")[1]
                        time = int(time_str.split(":")[0]) * 3600 + int(time_str.split(":")[1]) * 60 + int(
                            time_str.split(":")[2])
                        time = int(int(time) // time_format * time_format)
                        lng = line.split(",")[2]  # 经度
                        lat = line.split(",")[3]  # 纬度
                        # x, y = convert_latlon_to_xy(lng, lat)
                        # data = [time, id, x, y]
                        data = [time, id, lng, lat]
                        data = pd.DataFrame(pd.Series(data))
                        count += 1
                        df = pd.concat([df, data], axis=1)
                        if count % 1000 == 0:
                            print("write ", count , "times in file ", code)
                        # df[count] = data
    # output_orderByCar_sample_file.close()
    df = df.T
    print("df[0]:",df[0])
    df = df.sort_values(0, ascending=True)  # sorted by time
    df = df.T

    with open(output_orderByCar_sample_file_path, "w") as write_f:
        df.to_csv(write_f,header=False, index=False)

    print("finish write to ", output_orderByCar_sample_file_path)


if __name__ == "__main__":
    get_from_code2code()
    #get_data_in_frame(input_file="sample/22223_22224/true_pos_.csv",output_file="sample/22223_22224/sample_taxi_data.csv")
    # get_sample_file()
    # get_data_in_frame()
    # id_sort_in_format()
    ## time_sort_in_format()