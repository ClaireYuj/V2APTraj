#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   my_learning.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/11/11 19:41
------------      --------    -----------

"""
import math

import torch

from pyproj import Proj, transform, Transformer, CRS


def convert_latlon_to_xy(longitude, latitude):
    # 定义投影坐标系（WGS 84坐标系）

    # 转换经纬度为平面坐标
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395")
    print("transformer.transform(longitude, latitude):",transformer.transform(latitude,longitude))
    x, y = transformer.transform(latitude, longitude)
    x = float(str(x).split(".")[0][-5:] + "." +str(x).split(".")[1][:4])
    y = float(str(y).split(".")[0][-5:] + "." +str(y).split(".")[1][:4])

    return x, y

def millerToXY (lon, lat):
    xy_coordinate = []
    # 地球周长
    L = 6381372 * math.pi * 2
    # 平面展开，将周长视为X轴
    W = L
    # Y轴约等于周长一般
    H = L / 2
    # 米勒投影中的一个常数，范围大约在正负2.3之间
    mill = 2.3
    # 将经度从度数转换为弧度
    x = lon * math.pi / 180
    # 将纬度从度数转换为弧度
    y = lat * math.pi / 180
    # 这里是米勒投影的转换
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))
    # 这里将弧度转为实际距离 ，转换结果的单位是公里
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    xy_coordinate.append((int(round(x)), int(round(y))))


    return x, y



# 示例经纬度（深圳）
latitude = 22.555468
longitude = 114.14695

lat_2 = 22.555317
lng_2 = 114.145714

# 转换为平面坐标
x, y = convert_latlon_to_xy(longitude, latitude)
x2, y2 = convert_latlon_to_xy (lng_2, lat_2)

print(f"经度: {longitude}, 纬度: {latitude}")
print(f"平面坐标 X: {x}, Y: {y}")


print(f"平面坐标 X: {x2}, Y: {y2}")