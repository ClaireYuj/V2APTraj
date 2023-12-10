#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DataDivision.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/2/6 13:41
------------      --------    -----------

"""
import os


def sortDataOfFilesByTimeInSequence(unorderd_directory, orderd_directory):
    """
     将在unorder目录下的所有文件都按照时间顺序拍好
    :param upper_directory: 未按时间顺序排序文件存放的目录的上一级目录
    :return:
    """
    for filename in os.listdir(unorderd_directory):
        if filename.endswith('.txt') and filename.startswith("taxiCode_"):
            # code = int((filename.split(".")[0]).split("_")[1])
            # if code == 25303:
            print("Start sort" + filename + "! ")
            order_file = sortDataByTime(unorderd_directory, orderd_directory, filename)
            print("sort "+filename+"sucessfully!")
            order_file.close()

def sortDataByTime(unorderd_directory, orderd_diretory,filename):
    """
    对单个文件进行按时间排序
    :param unorderd_directory: 未按照时间顺序排序的文件存放的目录, 读取的文件目录
    :param orderd_directory: 按照时间顺序排序的文件存放的目录， 写入的文件目录
    :param filename: 需排序的文件名
    :return: order_file
    """
    file_path = os.path.join(unorderd_directory, filename)
    if not os.path.exists(orderd_diretory):
        # 如果目录不存在，则创建目录
        os.makedirs(orderd_diretory)
    with open(file_path, 'r') as file:
        info_list = []

        while True:
            line = file.readline()

            if line:
                info_list.append(line)
            else:
                break

    unique_info_list = deduplication(info_list)
    order_file_name = orderd_diretory + filename
    order_file = open(order_file_name, "w")
    unique_order_info_list = quick_sort_by_time_intervals(unique_info_list, time_index=1)
    # order_info_list = quick_sort(info_list, time_index=2)

    print("finish sort " + str(filename))

    # for line in order_info_list:
    for line in unique_order_info_list:
        order_file.write(line)
    print("finish write to new file "+str(order_file_name))
    return order_file


def quick_sort_by_time_intervals(info_list, time_index=1):
    """
    快排，对包含了时间信息的整个列表按照时间顺序排序
    :param info_list: 一个包含时间信息的列表，格式为如[[22223, 18520], [22225,78520]],不是时刻而是时间间隔
    :return:
    """
    length = len(info_list)
    if len(info_list) <= 1:
        return info_list
    # pivot_index = len(info_list) // 2
    pivot_index = length // 2
    pivot = int((info_list[pivot_index].split(",")[time_index]))
    left = [info_list[x] for x in range(0, length)
            if int((info_list[x].split(",")[time_index])) < pivot and x != pivot_index]
    middle = [info_list[pivot_index]]
    right = [info_list[x] for x in range(0, length)
             if int((info_list[x].split(",")[time_index])) >= pivot and x != pivot_index]

    return quick_sort_by_time_intervals(left, time_index=1) + middle + quick_sort_by_time_intervals(right, time_index=1)

def deduplication(info_list):

    # unique_info_list = list(set(info_list))
    unique_info_list = []
    for info in info_list:
        # print("Start read from unorder file " + str(info))
        if info not in unique_info_list:
            # print("Start write to unorder file " + str(info))
            unique_info_list.append(info)
    return unique_info_list



def quick_sort(info_list, time_index=1):
    """
    快排，对包含了时间信息的整个列表按照时间顺序排序
    :param info_list: 一个包含时间信息的列表，格式为如[[22223, 10:00:01], [22225,10:00:03]]
    :return:
    """
    length = len(info_list)
    if len(info_list) <= 1:
        return info_list
    # pivot_index = len(info_list) // 2
    pivot_index = length // 2
    pivot = int(str((info_list[pivot_index].split(",")[time_index]).split(":")[0]) + str((info_list[pivot_index].split(",")[time_index]).split(":")[1])
                + str((info_list[pivot_index].split(",")[time_index]).split(":")[2]))
    left = [info_list[x] for x in range(0, length)
            if int(str((info_list[x].split(",")[time_index]).split(":")[0]) + str((info_list[x].split(",")[time_index]).split(":")[1])
                   + str((info_list[x].split(",")[time_index]).split(":")[2])) < pivot and x != pivot_index]
    middle = [info_list[pivot_index]]
    right = [info_list[x] for x in range(0, length)
             if int(str((info_list[x].split(",")[time_index]).split(":")[0]) + str((info_list[x].split(",")[time_index]).split(":")[1])
                   + str((info_list[x].split(",")[time_index]).split(":")[2])) >= pivot and x != pivot_index]

    return quick_sort(left, time_index=2) + middle + quick_sort(right, time_index=2)



def divideDataByTime(filename, directory="processingData/rearrangeTaxiTime/hourAndMinute"):
    """
    基于时间对原数据划分
    :param filename: 原数据所在文件名
    :return:
    """
    file = open(filename, "r")

    generated_file_dic = {}

    """ h: hour, m: minute, m_1: minute-10"""
    for h in range(0, 24):
        if h < 10:
            h = str("0"+str(h))
        else:
            h = str(h)
        for m in range(10, 61, 10):
            if m == 10:
                m_1 = "00" + "-" +str(m)
            else:
                m_1 = str(m - 10) + "-" + str(m)
            m = str(m)
            string_h_m = h + "_" + m
            new_file = open(directory+"/TaxiData"+h+"_"+m_1+".txt", "w")
            generated_file_dic.update({string_h_m: new_file})
    print(generated_file_dic)

    while True:
        line = file.readline()
        if line:
            time = line.split(",")[1]
            hour = time.split(":")[0]
            min = int(time.split(":")[1])
            min = str((min // 10) * 10 + 10)
            selected_file = generated_file_dic.get(hour+"_"+min)
            selected_file.write(line)
        else:
            break
    for files in generated_file_dic.values():
        files.close()
    file.close()


def dataDivision(original_data_dir, filename,  divided_data_dir):
    """
    按照车辆代码对数据划分
    :param filename: 原数据所在文件名
    :return:
    """
    file = open(original_data_dir+filename, "r")
    if not os.path.exists(divided_data_dir):
        # 如果目录不存在，则创建目录
        os.makedirs(divided_data_dir)

    pre_code = 0
    # code file 的数字是源taxiCode文件中第一个车的代码
    code_file = open(divided_data_dir+"start"+".txt", "a")
    code_info_list = []
    line_count = 0
    while True:
        line = file.readline()
        if line:
            code = line.split(",")[1]
            len_line_split = line.split(",").__len__()  # get line split by ","
            for i in range(len_line_split // 4):
                try:
                    lat = float(line.split(",")[4 * i + 2])
                    lng = float(line.split(",")[4 * i + 3])
                    code = int((line.split(",")[4 * i + 0]).split(".")[0])
                    time = str((line.split(",")[4 * i + 1]).split(".")[0])
                    if code != pre_code:
                        code_file.close()
                        path = divided_data_dir + "taxiCode_" + str(code) + ".txt"
                        pre_code = code
                        code_file = open(path, "a")
                        print("Start write to unorder file "+str(code) +"......")
                        # code_info_list.append(str(code) + "," + str(time) + "," + str(lat) + "," + str(lng)+'\n')
                    code_file.write(str(code) + "," + str(time) + "," + str(lat) + "," + str(lng)+'\n')

                except Exception as result:
                    print(result)
        else:
            break
    print("finish get code info list")
    # unique_code_info_list = deduplication(code_info_list)
    #
    # for line in unique_code_info_list:
    #     code_file.write(line)

def dataDivisionByTaxiCode(original_data_dir, filename,  divided_data_dir):
    """
    按照车辆代码对数据划分
    :param filename: 原数据所在文件名
    :return:
    """
    file = open(original_data_dir+filename, "r")
    if not os.path.exists(divided_data_dir):
        # 如果目录不存在，则创建目录
        os.makedirs(divided_data_dir)

    pre_code = 0
    # code file 的数字是源taxiCode文件中第一个车的代码
    code_file = open(divided_data_dir+"start"+".txt", "a")
    code_info_list = []
    line_count = 0
    while True:
        line = file.readline()
        if line:
            # code = line.split(",")[1]


                len_line_split = line.split(",").__len__()  # get line split by ","
                # for i in range(len_line_split // 4):
                #
                #     try:
                        # lat = float(line.split(",")[4 * i + 2])
                        # lng = float(line.split(",")[4 * i + 3])
                        # code = int((line.split(",")[4 * i + 0]).split(".")[0])
                        # time = str((line.split(",")[4 * i + 1]).split(".")[0])
                try:

                    if line_count % 4 == 0:
                        code = int((line.split("\n")[0]).split(".")[0])
                    elif line_count % 4 == 1:
                        time = str((line.split("\n")[0]).split(".")[0])
                    elif line_count % 4 == 2:
                        lat = float(line.split("\n")[0])
                    elif line_count % 4 == 3:
                        lng = float(line.split("\n")[0])
                    if line_count != 0 and line_count % 4 == 0:
                        if code != pre_code:
                            code_file.close()
                            path = divided_data_dir + "taxiCode_" + str(code) + ".txt"
                            pre_code = code
                            code_file = open(path, "a")
                            print("Start write to unorder file "+str(code) +"......")
                        # code_info_list.append(str(code) + "," + str(time) + "," + str(lat) + "," + str(lng)+'\n')
                        code_file.write(str(code) + "," + str(time) + "," + str(lat) + "," + str(lng)+'\n')
                    line_count += 1
                except Exception as result:
                    print(result)
        else:
            break
    print("finish get code info list")

def dataDivisionByTime(original_data_dir, filename,  divided_data_dir):
    """
    按照车辆代码对数据划分
    :param filename: 原数据所在文件名
    :return:
    """
    file = open(original_data_dir+filename, "r")
    if not os.path.exists(divided_data_dir):
        # 如果目录不存在，则创建目录
        os.makedirs(divided_data_dir)

    pre_time_range_index = 0
    # original time file is start.txt file
    time_file = open(divided_data_dir+"start"+".txt", "a")
    code_info_list = []
    line_count = 0
    while True:
        line = file.readline()
        if line:
                try:
                    if line_count % 4 == 0:
                        code = int((line.split("\n")[0]).split(".")[0])
                    elif line_count % 4 == 1:
                        time = str((line.split("\n")[0]).split(".")[0])
                        time_range_index = int(time) // 3600  # let 1 hour as the borderline
                    elif line_count % 4 == 2:
                        lat = float(line.split("\n")[0])
                    elif line_count % 4 == 3:
                        lng = float(line.split("\n")[0])

                    if line_count != 0 and line_count % 4 == 0:
                        if time_range_index != pre_time_range_index:
                            time_file.close()
                            path = divided_data_dir + "timeRange_" + str(time_range_index)+"_"+ str((time_range_index+1)) + "_clocktxt"
                            pre_time_range_index = time_range_index
                            time_file = open(path, "a")
                            print("Start write to unorder file "+str(time_range_index) +"......")
                        # code_info_list.append(str(code) + "," + str(time) + "," + str(lat) + "," + str(lng)+'\n')
                        time_file.write(str(code) + "," + str(time) + "," + str(lat) + "," + str(lng)+'\n')
                    line_count += 1
                except Exception as result:
                    print(result)
        else:
            break
    print("finish get time info list")


def divideTrueAndPredDataByCode(carNum):
    taxi_num_dir = "taxi_"+str(carNum) + '/'
    dataDivisionByTaxiCode("./savedata", "/" + taxi_num_dir + "GATraj/" + "true" + "_trajectory.csv",
                           "./processingData/" + taxi_num_dir + "/" + "true" + "/code_unorder/")
    new_true_data_directory = "./processingData/" + taxi_num_dir + "/" + "true" "/"

    # 把unorder目录下所有文件都进行时间排序，输出到orderbytime目录下, 若是直接用taxi_data，已经排序好了，暂时不用
    sortDataOfFilesByTimeInSequence(new_true_data_directory + "unorder/", new_true_data_directory + "code_order/")
    dataDivisionByTaxiCode("./savedata", "/" + taxi_num_dir + "GATraj/" + "predicted" + "_trajectory.csv",
                           "./processingData/" + taxi_num_dir + "/" + "predicted" + "/code_unorder/")
    new_pred_data_directory = "./processingData/" + taxi_num_dir + "/" + "predicted" "/"

    # 把unorder目录下所有文件都进行时间排序，输出到orderbytime目录下, 若是直接用taxi_data，已经排序好了，暂时不用
    sortDataOfFilesByTimeInSequence(new_pred_data_directory + "code_unorder/", new_pred_data_directory + "code_order/")


def divideTrueAndPredDataByTime(carNum):
    taxi_num_dir = "taxi_"+str(carNum) + '/'
    dataDivisionByTime("./savedata", "/" + taxi_num_dir + "GATraj/" + "true" + "_trajectory.csv",
                           "./processingData/" + taxi_num_dir + "/" + "true" + "/time_unorder/")
    new_true_data_directory = "./processingData/" + taxi_num_dir + "/" + "true" "/"

    # 把unorder目录下所有文件都进行时间排序，输出到orderbytime目录下, 若是直接用taxi_data，已经排序好了，暂时不用
    sortDataOfFilesByTimeInSequence(new_true_data_directory + "time_unorder/", new_true_data_directory + "time_order/")
    dataDivisionByTaxiCode("./savedata", "/" + taxi_num_dir + "GATraj/" + "predicted" + "_trajectory.csv",
                           "./processingData/" + taxi_num_dir + "/" + "predicted" + "/time_unorder/")
    new_pred_data_directory = "./processingData/" + taxi_num_dir + "/" + "predicted" "/"

    # 把unorder目录下所有文件都进行时间排序，输出到orderbytime目录下, 若是直接用taxi_data，已经排序好了，暂时不用
    sortDataOfFilesByTimeInSequence(new_pred_data_directory + "time_unorder/", new_pred_data_directory + "time_order/")


if __name__ == "__main__":

    # divideTrueAndPredDataByCode(500)
    divideTrueAndPredDataByTime(500)
    # taxiCode目录下要有unorder和sortByTimeInSequence两个目录
    # 目录结构"./processingData/rearrangeTaxiTime/taxiCode/unorder"
    # 目录结构 "./processingData/rearrangeTaxiTime/taxiCode/sortByTimeInSequence"

    # sortDataByTime(new_data_directory+"unorder/", new_data_directory+"orderByTime/", "taxiCode_22223.txt")

