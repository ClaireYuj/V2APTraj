import math
import numpy as np
import folium
import webbrowser
from folium import plugins
import pandas as pd
from folium.plugins import HeatMap
import time
import pyproj

import DataDivision

# 地球半径

# EARTH_RADIUS = 6378.137
EARTH_RADIUS = 6378.137
TRAIN_DATA_ROOT = "./data/train/prediction_train/"
TEST_DATA_ROOT = "./data/test/prediction_test/"


processingData_root = "./processingData/" # 存放一些中间数据的目录

SUB_AREA_LENGTH = 1000
SUB_AREA_WIDTH = 1000


def rad(d):
    return d * math.pi / 180.0


def radToDegree(rad):


    return rad * 180.0 / math.pi


# double s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a / 2), 2) +
#                 Math.cos(radLat1) * Math.cos(radLat2) * Math.pow(Math.sin(b / 2), 2)));
#         s = s * EARTH_RADIUS;
#         s = new BigDecimal(s).setScale(3, BigDecimal.ROUND_HALF_UP).doubleValue();
#         return s;
def calDis(lat1, lng1, lat2, lng2):
    # 输入两点的lat/lng，得到两点的距离(m)
    radLat1, radLat2 = rad(lat1), rad(lat2)
    radLng1, radLng2 = rad(lng1), rad(lng2)
    lat_df = radLat1 - radLat2
    lng_df = radLng1 - radLng2
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(lat_df / 2), 2)
                                + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(lng_df / 2), 2)))

    s = s * EARTH_RADIUS * 1000
    return s



def outputDisToFile(taxiData, code_time_data, isOrderByTime, true_or_pred="true"):
    """
    把两点间的距离输出到./processingData目录下的distance.txt文件
    :param taxiData: 车辆经纬度列表
    :param code_time_data: 车辆代码与时间列表
    :param isOrderByTime: 是否按时间顺序拍好的bool变量
    :return:
    """
    if isOrderByTime:
        isOrderString = str("orderedByTime")
    else:
        isOrderString = str("unorderedByTime")

    dis_file = open(processingData_root+"distance_"+str(code_time_data[0][0])+"_"+isOrderString+".txt", "w")
    lenOfData = len(taxiData)
    for i in range(1, lenOfData):
        lat1, lng1 = taxiData[i-1]
        lat2, lng2 = taxiData[i]
        dis = calDis(lat1, lng1, lat2, lng2)
        dis_file.write("code: "+str(code_time_data[0][0])+"  time: "+str(code_time_data[i-1][1])+"-"+str(code_time_data[i][1])
                       +"  distance: "+str(dis)+"m   from lat:"+str(lat1)+" lng:"+str(lng1)+" to lat:"+str(lat2)+" lng: "+str(lng2)+"\n")

    dis_file.close()


def calAvgTimeInterval(code_time_data):
    """
    计算平均时间间隔
    :param code_time_data: [[22223, 00:00:01],[22223,10:02:01]
    :return:
    """
    avg_time_list = []
    length = len(code_time_data)
    count = 0
    # time_file = open("./processingData/time_interval_" + str(code_time_data[0][0]) + ".txt", "w")
    sum_time_interval = 0
    for i in range(1, length):
        time_pre_str = code_time_data[i-1][1]
        time_cur_str = code_time_data[i][1]
        hour_pre = time_pre_str.split(":")[0]
        hour_cur = time_cur_str.split(":")[0]
        time_pre = int(time_pre_str.split(":")[0])*3600+int(time_pre_str.split(":")[1])*60+int(time_pre_str.split(":")[2])
        time_cur = int(time_cur_str.split(":")[0])*3600+int(time_cur_str.split(":")[1])*60+int(time_cur_str.split(":")[2])
        if (time_cur - time_pre) < 1800: # 只读时间间隔在30min以内的
        # if hour_cur == hour_pre: # 是同一个小时
            sum_time_interval += (time_cur - time_pre)
            avg_time_list.append((time_cur-time_pre))
            count += 1

        #     print("code_time_data[i-1][1]:",code_time_data[i-1][1]," code_time_data[i][1]",code_time_data[i-1][1])
        # else:
        #     print("pre:", code_time_data[i - 1][1],"tine:pre: ",time_pre, " cur", code_time_data[i - 1][1], "time_cur:", time_cur)

    if length == 0 or length == 1 or count == 0: # count == 0 说明数据太过于分散
        return 0, []
    elif length > 1:
        sum_time_interval /= count
    return sum_time_interval, avg_time_list



def calLngOfVertexByDistance(lat0, lng0, s):
    """
    在确定区域位置时，对于相同纬度的顶点，要确定其经度
    ，通过获得另一个顶点的经纬度，以及两点间距离，最终求得该点的经度
    :param lat0:
    :param lng0:
    :param s:
    :return:
    """
    lat0 = rad(lat0)
    lng0 = rad(lng0)
    lng1 = rad(90)

    s = s / (EARTH_RADIUS * 1000)
    try:
        lng1 = math.acos(1 - (2 * math.pow(math.sin(s/2), 2)) / (math.pow(math.cos(lat0), 2))) + lng0
    except Exception as result:
        print(result)


    return radToDegree(lng1)




def calLatOfVertexByDistance(lat0, lng0, s):
    """
    在确定区域位置时，对于相同经度度的顶点，要确定其纬度
    ，通过获得另一个顶点的经纬度，以及两点间距离，最终求得该点的纬度
    :param lat0:
    :param lng0:
    :param s:
    :return:
    """
    lat0 = rad(lat0)
    lng0 = rad(lng0)
    s = s / (EARTH_RADIUS * 1000)
    lat1 = s + lat0

    return radToDegree(lat1)



def calRectArea(lat1, lng1, lat3, lng3):
    """
    求矩形面积，需要对角线两点的经纬度
    :param lat1:
    :param lng1:
    :param lat3:
    :param lng3:
    :return:
    """
    lat2 = lat1
    lng2 = lng3
    lat4 = lat3
    lng4 = lng1
    print("lat1:", lat1, "lng1:", lng1, "lat4:", lat4, "lng4:", lng4)
    print("lat4:", lat4, "lng4:", lng4, "lat3:", lat3, "lng3:", lng3)

    print("calDisL: ", calDis(89.7, 88.6, 89.7, 18.5))

    # 获得四个点的坐标
    width = calDis(lat1, lng1, lat4, lng4)
    length = calDis(lat4, lng4, lat3, lng3)
    print("width:", width, "length:", length)
    return length, width



def getVertex(lng_and_lat_data):
    """
     从所有经纬度中获得可以确定边界范围的两点（对角线）的经纬度，确定四个顶点
    :param lng_and_lat_data:
    :return: lat_max, lng_max, lat_min, lng_min
    """
    lat_min, lat_max = lng_and_lat_data[0][0], lng_and_lat_data[0][0]
    lng_min, lng_max = lng_and_lat_data[0][1], lng_and_lat_data[0][1]
    for data_ in lng_and_lat_data:
        if data_[0] > lat_max:
            lat_max = data_[0]
        elif data_[0] < lat_min:
            lat_min = data_[0]
        if data_[1] > lng_max:
            lng_max = data_[1]
        elif data_[1] < lng_min:
            lng_min = data_[1]

    print("min_lat:", lat_min, "min_lng:", lng_min, "max_lat", lat_max, "max_lng:", lng_max)
    return lat_max, lng_max, lat_min, lng_min

def dynamicAreaParition(datalist, true_or_pred="true", ordered_by_time_or_code="time"):
    """
    基于传入的车辆的经纬度动态划分区域,（区域的位置会因为车辆的经纬度而改变）
    传入的是经纬度的list data= [[lat1, lng1], [lat2, lng2]]
    可以利用全局变量 SUB_AREA_LENGTH 和 SUB_AREA_WIDTH控制区域大小
    返回画好的区域的字典数组，可以在FinialCarInArea.txt文档中查看
    :param datalist: 车辆经纬度的列表[[lat1,lng1],[lat2,lng2]
    :return: region_map: writen in Final Car text
    """

    vertex = getVertex(datalist)
    total_length, total_width = calRectArea(vertex[0], vertex[1], vertex[2], vertex[3])
    MIN_LAT = vertex[2]
    MIN_LNG = vertex[3]
    print("total length:", total_length, "total width:", total_width)

    min_lat = MIN_LAT
    min_lng = MIN_LNG

    lat_list = [MIN_LAT]
    lng_list = [MIN_LNG]
    test_area_file = open(processingData_root+true_or_pred+"/"+ordered_by_time_or_code+"_order"+"/testAreaPart.txt", "w")
    region_map = []
    len_slide = int(total_length // SUB_AREA_LENGTH)
    if total_length % SUB_AREA_LENGTH != 0:
        len_slide += 1
    width_slide = int(total_width // SUB_AREA_WIDTH)

    if total_width % SUB_AREA_WIDTH != 0:
        width_slide += 1

    print("len_slide:", len_slide, "Width:", width_slide)
    for i in range(len_slide):
        for j in range(width_slide):
            region_map.append({"area_index": str(i)+"-"+str(j), "min_lat": min_lat,
                        "min_lng": min_lng, "max_lat": calLatOfVertexByDistance(min_lat, min_lng, SUB_AREA_WIDTH),
                        "max_lng": calLngOfVertexByDistance(min_lat, min_lng, SUB_AREA_LENGTH), "carNum": 0})
            test_area_file.write(str({"area_index": str(i) + "-" + str(j), "min_lat": min_lat,
                         "min_lng": min_lng, "max_lat": calLatOfVertexByDistance(min_lat, min_lng, SUB_AREA_WIDTH),
                         "max_lng": calLngOfVertexByDistance(min_lat, min_lng, SUB_AREA_LENGTH), "carNum": 0})+"\n")
            min_lat = calLatOfVertexByDistance(min_lat, min_lng, SUB_AREA_WIDTH)
            if len(lat_list) <= width_slide:
                lat_list.append(min_lat)
        min_lat = MIN_LAT
        min_lng = calLngOfVertexByDistance(min_lat, min_lng, SUB_AREA_LENGTH)
        lng_list.append(min_lng)


    test_area_file.write("lat list: "+str(lat_list)+"\nlng list: "+str(lng_list))
    print("finish divide the lat and lng.......")

    car_record_file = open(processingData_root+true_or_pred+"/"+ordered_by_time_or_code+"_order"+"/CarRecord.txt", "w")
    # car_record_file.write("this file ordered by "+ordered_by_time_or_id)
    for data in datalist:
        try:
            lat_index, lng_index = findAreaIndexofCar(lat_list, lng_list, data[0], data[1])
            car_index = lng_index * width_slide + lat_index

            region_map[car_index]["carNum"] += 1

            car_record_file.write("car_lat: "+str(data[1])+"car_lng: "+str(data[0])+" area_code: "+str(lat_index)+"-"+str(lng_index)+"\n")
        except Exception as e:
            print(e)
            print("data", data)
            print("map_len:", len(region_map), " car_index:", car_index, "lng_index", lng_index, "lat_index:", lat_index)
        # finally:
        #     print("data", data)
        #     print("map_len:", len(map)," car_index:", car_index, "lng_index",lng_index,"lat_index:", lat_index)
        #     print(" map[car_index][\"carNum\"]: ", map[car_index]["carNum"],)
    print("finish allocate the car in area..........")
    final_area_recode_file = open(processingData_root+true_or_pred+"/"+ordered_by_time_or_code+"_order"+"/FinalCarInArea.txt", "w")
    lat_list_by_area_col_id = [] # record the list of each col, because the area_index format as '0-1' ,'1-1', the list record the sum the lat start with['0-', '1-']
    lng_list_by_area_col_id = []
    last_area_index_col = -1
    for m in region_map:
        if m["carNum"] > 0:
            m_area_index_col = int(m["area_index"].split("-")[0])
            print("last index:",last_area_index_col," m_area_index:", m_area_index_col)
            if m_area_index_col != last_area_index_col:
                # lat_list_by_area_col_id.append(float(m["min_lat"]) * int(m['carNum']))
                # lng_list_by_area_col_id.append(float(m["min_lng"]) * int(m['carNum']))

                while m_area_index_col != len(lat_list_by_area_col_id) - 1:
                    print("lat_list_by_area_col_id:",lat_list_by_area_col_id)
                    print("m_area_index:",m_area_index_col, " len:",len(lat_list_by_area_col_id))
                    lat_list_by_area_col_id.append(0)
                    lng_list_by_area_col_id.append(0)
                lat_list_by_area_col_id.append(float(m["min_lat"]) * int(m['carNum']))
                lng_list_by_area_col_id.append(float(m["min_lng"]) * int(m['carNum']))
            else:
                lat_list_by_area_col_id[m_area_index_col] += float(m["min_lat"]) * int(m['carNum'])
                lng_list_by_area_col_id[m_area_index_col] += float(m["min_lng"]) * int(m['carNum'])
            final_area_recode_file.write(str(m)+"\n")
            last_area_index_col = m_area_index_col
    final_area_recode_file.write("\nlat list:"+str(lat_list_by_area_col_id))
    final_area_recode_file.write("\nlng list:"+str(lng_list_by_area_col_id))
    print("finish write the car in file......")
    final_area_recode_file.close()
    car_record_file.close()
    test_area_file.close()
    return region_map

def trueInPredArea():
    """
    how many true car in the pred area?

    :return:
    """


def drawCarDensity(region_map, schools_map, regionHtmlPath, true_or_pred="true"):
    """
    在地图上标出区域的范围， 并显示该区域内的车辆数
    green is true, red is predicted
    :param region_map:  dynamicAreaParition的返回值，形如[{"area index": 1, "min_lat":10, "min_lng":10, "max_lat":20, "max_lng":20, "carNum":30},{...},{...}]
    :param schools_map: drawScatter后的返回值
    :param regionHtmlPath: 存放画车辆密度的html路径
    :return:
    """
    print("Start drawing the density figure.......")
    for region in region_map:
        if region["carNum"] != 0:
            min_lat = region["min_lat"]
            min_lng = region["min_lng"]
            max_lat = region["max_lat"]
            max_lng = region["max_lng"]

            bound = [[min_lat, min_lng], [max_lat, max_lng]]
            rect_color = "green" if true_or_pred == "true" else "red"
            folium.Rectangle(
                bound,
                # popup="</br>car num:"+str(region["carNum"])+"</br>",
                tooltip="</br>"+true_or_pred+" car num:"+str(region["carNum"])+"</br>",
                fill_color = rect_color,
                fill=True,
                color=rect_color
            ).add_to(schools_map)
            # print(" draw one : min_lat:", min_lat, "min_lng:", min_lng, "max_lat:", max_lat, "max_lng:", max_lng)

    schools_map.save(regionHtmlPath)


def drawCartrajectory(taxiData, code_time_list, map ,new_map_path,true_or_pred="true", ordered_by_time_or_code="time"):
    """
    画出行车轨迹
    :param taxiData: 包含经纬度的列表，形如[[lat1, lng1], [lat2,lng2]
    :param code_time_list: 包含车辆编码和时间的列表，如[[22223,00:00:01],[22223,10:02:01]
    :param map: 已经标号点的地图，foilum.Map变量，（是drawscatter或者drawscatterByPartition的返回值）
    :param new_map_path: 存放行车轨迹的路径
    :return:
    """
    print("Start draw trajectory.....")
    car_trjectory_file = open(processingData_root+true_or_pred+"/"+ordered_by_time_or_code+"_order/"+"/car_traj_record.txt","w")
    locations = [taxiData[0]]
    lenOfData = len(taxiData)
    for i in range(1, lenOfData):
        point = taxiData[i]
        locations.append(point)
        data_0 = code_time_list[i-1]
        data_1 = code_time_list[i]
        taxi_info = "code: "+str(data_0[0])+" time: "+str(data_0[1])+"-"+str(data_1[1])
        car_trjectory_file.write(true_or_pred + " code: "+str(data_0[0])+" time: "+str(data_0[1])+"-"+str(data_1[1])+" location:"+str(locations)+"\n")
        # 包含车辆信息
        line_color = "blue" if true_or_pred == "true" else "red"
        polyline = folium.PolyLine(locations=locations, color=line_color, weight=2.5, opacity=0.5,
                        arrow_style='fancy', popup="<br>"+taxi_info+"</br>").add_to(map)

        # 显示行车方向
        folium.plugins.PolyLineTextPath(
            polyline=polyline,
            align='center',
            attributes={"fill": "yellow"},
            weight=2,
            opacity=1,
            font_family='Arial, sans-serif',
            font_size=10,
            font_weight='bold',
            text='▶'
        ).add_to(map)

        locations.pop(0)
    car_trjectory_file.close()
    map.save(new_map_path)
    print("finish drawing trajectory...")




def findAreaIndexofCar(lat_list, lng_list, car_lat, car_lng):
    """
    获得车辆的所在区域的index
    :param lat_list:
    :param lng_list:
    :param car_lat:
    :param car_lng:
    :return:
    """
    lat_index = len(lat_list)
    lng_index = len(lng_list)
    for i in range(len(lat_list)):
        if car_lat < lat_list[i]:
            lat_index = i-1
            break
    for j in range(len(lng_list)):
        if car_lng < lng_list[j]:
            lng_index = j-1
            break
    return lat_index, lng_index




# 显示在之前的地图插件上，看看具体位置
def drawscatt(data, mapHtmlpath, true_or_pred="true"):
    print("Start draw........")
    schools_map = folium.Map(location=data[0], zoom_start=10,
                             tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}",
                             attr="&copy; <a href='http://ditu.amap.com/'>高德地图</a>"
                             # title="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}",
                             # # attr="&copy; <a href='http://ditu.amap.com/'>高德地图</a>"
                             # tiles='OpenStreetMap'
                             )
    marker_cluster = plugins.MarkerCluster().add_to(schools_map)

    count = 0
    start_time = time.time()
    # 标注数据点：
    for j in range(len(data)):
        folium.Marker(data[j]).add_to(
            marker_cluster)
        if j % 100000 == 0:
            end_time = time.time()
            count += 1
            print("draw 10000 x ", count," points, time cost:",end_time - start_time,"s")
            start_time = time.time()

    schools_map.save(mapHtmlpath)  # 将地图保存在电脑文件里
    return schools_map
    # webbrowser.open(mapHtmlpath)


def createMapCanvas(taxiData, create_or_not=0):
    """
    :param: taxiData: to set the central of the map
    :param: create_or_not: if create_or_not=0, not be created before,
            if create_or_not=1, have been created before, ignore this operation
    """

    schools_map = folium.Map(location=taxiData[0], zoom_start=10,
                             tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}",
                             attr="&copy; <a href='http://ditu.amap.com/'>高德地图</a>",

                             )
    return schools_map

def drawscattByPartition(taxiData, schools_map,  mapHtmlpath, true_or_pred="true"):
    """
    难以直接把所有的点一次性画出来， 所以分步画
    :param taxiData: 包含经纬度的列表，形如[[lat1, lng1], [lat2, lng2]
    :param mapHtmlpath: 输出html地址
    :param true_or_pred: if true: blue, if pred: red
    :return:
    """
    print("start drawing partition of "+true_or_pred+".......")
    lenOfData = len(taxiData)
    size = 100000
    count = 0
    marker_color = "blue" if true_or_pred == "true" else "red"
    icon = folium.Icon(color=marker_color)


    # schools_map = folium.Map(location=taxiData[0], zoom_start=10,
    #                          tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}",
    #                          attr="&copy; <a href='http://ditu.amap.com/'>高德地图</a>",
    #
    #                          )

    # marker_cluster = plugins.MarkerCluster().add_to(schools_map)
    marker_cluster = plugins.MarkerCluster(marker_color=marker_color).add_to(schools_map)
    start_time = time.time()
    index = 0
    while index > lenOfData:
        try:

            subData = taxiData[index:index+size]

            for j in range(len(subData)):
                string_car = str(subData[j])
                info = "code: "+string_car.split(",")[0]+" time: "+string_car.split(",")[1]
                folium.Marker(subData[j],
                              icon=icon,
                              popup="</br>"+info+"</br>",
                              marker_color=marker_color
                              ).add_to(
                    marker_cluster)

        except Exception as result:
            print("Exception: ", result)
        count += 1
        index += size
        if (count % 10 == 0):
            end_time = time.time()
            print("Draw", size, " x ", count, " points successfully, time cost: ", end_time - start_time,"s")
            start_time = time.time()
            schools_map.save(mapHtmlpath)
    for j in range(len(taxiData)):
        folium.Marker(taxiData[j]).add_to(
            marker_cluster)

    schools_map.save(mapHtmlpath)
    return schools_map

def readLatAndLngFromFile(filename):
    """
    从文件中读取经纬度
    :param filename:
    :return:
    """
    taxiData = []
    code_time_list = []
    lineSet = set()
    count = 0
    file = open(filename, "r")
    while True:
        line = file.readline()
        # code = int(line.split(",")[0])
        # code_diff = 36951 - code
        # if not line or code_diff < 0:
        #         break
        # elif code_diff > 0:
        if line:

            # len_line_split = line.split(",").__len__()  # get line split by ","
            # for i in range(len_line_split // 4):
            line = line.strip()
            if line in lineSet:
                continue
            lineSet.add(line)
            count += 1
            try:
                lat = float(line.split(",")[2])
                lng = float(line.split(",")[3])
                code = int(line.split(",")[0].split(".")[0])
                time = str(line.split(",")[1].split(".")[0])
                # lat = float(line.split(",")[4 * i + 2])
                # lng = float(line.split(",")[4 * i + 3])
                # code = int((line.split(",")[4 * i + 0]).split(".")[0])
                # time = str((line.split(",")[4 * i + 1]).split(".")[0])

            except Exception as result:
                print("count:", count, "line", line)
                print(result)
            if 0 < lat < 90 and 0 < lng < 180:
                taxiData.append([lat, lng])
                code_time_list.append([code, time])
            if count % 1000000 == 0:
                print("num:", count,  " lat:", lat, " ,lng:", lng)
        else:
            break
    print("finish read")
    file.close()
    return taxiData, code_time_list


def xyz_to_latlng(x, y, z):
    """
    将大地坐标系下的xyz坐标转化为经纬度坐标
    :param x:
    :param y:
    :param z:
    :return: lat, lng
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    lng = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e * e))
    for i in range(10):
        N = a / math.sqrt(1 - e * e * math.sin(lat) * math.sin(lat))
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - e * e * N / (N + h)))
    lat = math.degrees(lat)
    lng = math.degrees(lng)
    return lat, lng


def readXYZFromFile(filename):
    """
    从文件中读取XYZ坐标， 输出车辆的经纬度到./processingData/lat_and_lng.txt中
    :param filename:
    :return:
    """
    file_record = open(processingData_root+"lngAndLat.txt", "w")
    taxiData = []
    count = 0
    file = open(filename, "r")
    while True:
        line = file.readline()
        if line:
            code = int(line.split(" ")[0])

            try:
                X = float(line.split(" ")[3])
                Y = float(line.split(" ")[4])
                Z = float(line.split(" ")[5])
                lat, lng = xyz_to_latlng(X, Y, Z)

                file_record.write("lng: "+str(lng)+"  lat: "+str(lat)+"  X: "+str(X)+"  Y: "+str(Y)+"  Z: "+str(Z)+"\n")
                if 0 < lat < 90 and 0 < lng < 180:
                    taxiData.append([float(lat), float(lng)])
                    count += 1

                if count % 100000 == 0:
                    print("num:", count, " code:", code, " lat:", lat, " ,lng:", lng)
                #
                # print("X:", X, "Y:", Y, "Z:", Z, "lat:", lat, "lng:", lng)
                # print("R:", math.sqrt(math.pow(math.sqrt(X * X + Y * Y), 2) + Z * Z))

            except Exception as result:
                print("count:", count, "line", line)
                print(result)

        else:
            break
    print("finish read")
    file_record.close()
    return taxiData




def sortAreaByCarNum(carInAreaFilename, carInAreaSortByCarNumfilename):
    """
    按照车辆数量对车辆进行排序
    :param carInAreaFilename:
    :param carInAreaSortByCarNumfilename:
    :return:
    """
    carInAreaFile = open(carInAreaFilename, "r")
    carInAreaSortByCarNumFile = open(carInAreaSortByCarNumfilename, "w")

    infoList = []

    while True:
        line = carInAreaFile.readline()
        # print(type(line))
        if line and line.startswith("{'area_index'"):
            carNum = int(((line.split(",")[5]).split(":")[1]).split("}")[0])
            if carNum > 0:
                infoList.append(line)
        else:
            break
    min_car_num = int(((infoList[0].split(",")[5]).split(":")[1]).split("}")[0])

    print("Start sorting........")
    for i in range(1, len(infoList)):
        carNum_i = int(((infoList[i].split(",")[5]).split(":")[1]).split("}")[0])
        for j in range(len(infoList)-1):
            carNum_j = int(((infoList[j].split(",")[5]).split(":")[1]).split("}")[0])
            if carNum_i > carNum_j:
                tmp = infoList[i]
                infoList[i] = infoList[j]
                infoList[j] = tmp
    print("finish sorrting........")
    for line in infoList:
        carInAreaSortByCarNumFile.write(line)
    carInAreaSortByCarNumFile.close()
    carInAreaFile.close()

def showTaxiDataByCarCode(code, dataDiviedFlag=True, carNum='500'):
    """
    show the region, distribution and trajectory of car by code
    :param code: code of visulaized car
    :param dataDiviedFlag: is the records divided
    :param carNum: the training times -- Distinguishing on filename
    :return:
    """
    if not dataDiviedFlag:
        DataDivision.divideTrueAndPredDataByCode(carNum)

    car_code = str(code)
    true_or_pred = "true"
    mapHtmlpath = "result_data/" + car_code +"_train_" + carNum +"_times" +".html"
    regionHtmlPath = "result_data/" +   car_code + "_region_" + "train_" + carNum +"_times" + ".html"
    trajectoryPath = "result_data/" +   car_code + "_trajectory_" + "train_" + carNum +"_times" + ".html"
    for i in range(2):
        true_or_pred = "predicted" if true_or_pred == "true" else "true"
        ordered_file_directory = "./processingData/taxi_" + carNum + '/' + true_or_pred + '/code_order/'
        # mapHtmlpath = "result_data/" + true_or_pred + "_" + car_code + ".html"
        # regionHtmlPath = "result_data/" + true_or_pred + "_region_" + car_code + ".html"
        # trajectoryPath = "result_data/" + true_or_pred + "_trajectory_" + car_code + ".html"
        taxiData, code_time_list = readLatAndLngFromFile(
            ordered_file_directory + 'taxiCode_' + str(car_code) + '.txt')  # 读取经纬度
        if i == 0:
            map = createMapCanvas(taxiData)

        map = drawscattByPartition(taxiData, map, mapHtmlpath)  # 画出车辆经纬度图
        region = dynamicAreaParition(taxiData,true_or_pred, "code")     # 划分区域
        drawCarDensity(region, map, regionHtmlPath, true_or_pred) # 画出车辆密度图
        drawCartrajectory(taxiData, code_time_list, map, trajectoryPath,true_or_pred,"code")  # 画出车辆轨迹
        outputDisToFile(taxiData, code_time_list, True, true_or_pred)  # 输出两点间距离,输出到processingData目录下
    # print("The avg time interval is ", calAvgTimeInterval(code_time_list)[0],"s") # 计算平均时间

    # sortAreaByCarNum("./processingData/FinalCarInArea.txt", "./processingData/FinalCarInAreaWithoutZero.txt")

def showTaxisDataBetweenTime(start_time, end_time, dataDiviedFlag=True, carNum="500"):
    start_time = str(start_time)
    end_time = str(end_time)
    if not dataDiviedFlag:
        DataDivision.divideTrueAndPredDataByCode(carNum)

    true_or_pred = "true"
    mapHtmlpath = "result_data/" + start_time+ "_"+ end_time +"_train_" + carNum +"_times" +".html"
    regionHtmlPath = "result_data/" + start_time+ "_"+ end_time + "_region_" + "train_" + carNum +"_times" + ".html"

    for i in range(2):
        true_or_pred = "predicted" if true_or_pred == "true" else "true"
        ordered_file_directory = "./processingData/taxi_"+carNum+"/"+ true_or_pred + "/"
        taxiData, code_time_list = readLatAndLngFromFile(
            ordered_file_directory + "time_order/timeRange_" + str(start_time)+ "_"+ str(end_time) + '_clock.txt')  # 读取经纬度

        if i == 0:
            map = createMapCanvas(taxiData)

        map = drawscattByPartition(taxiData, map, mapHtmlpath, true_or_pred)  # 画出车辆经纬度图
        region = dynamicAreaParition(taxiData,true_or_pred,"time")     # 划分区域
        drawCarDensity(region, map, regionHtmlPath, true_or_pred) # 画出车辆密度图
        sortAreaByCarNum(processingData_root+true_or_pred+"/time_order/FinalCarInArea.txt", processingData_root+true_or_pred+"/time_order/FinalCarInAreaWithoutZero.txt")
# 计算gcj中两点的距离
# 将点显示在高德地图中，已经验证符合实际位置，可以使用


def performance_evaluation(final_area_file_dir, ordered_by_time_or_code="time"):
    """
    1. cal the area offset
    :return:
    """

    # cal the area offset


    true_final_coor_list = {"lat":[], "lng":[]}
    pre_final_coor_list = {"lat":[], "lng":[]}


    lat_offset_list = []
    lng_offset_list = []
    pre_list_len = 0
    true_list_len = 0
    with open(final_area_file_dir + "true/"+ordered_by_time_or_code+"_order"+"/FinalCarInArea.txt", "r") as true_final_area_file:

        while True:
            line = true_final_area_file.readline()
            print(line)
            if line.startswith("lat list"):
                tmp_lng_str = (line.split("lat list:[")[1]).split("]")[0]
                tmp_lng_list = tmp_lng_str.split(",")
                true_final_coor_list["lat"] = tmp_lng_list
                print("true_final_lat:", true_final_coor_list["lat"])
                true_list_len += 1
            elif line.startswith("lng list"):
                tmp_lat_str = (line.split("lng list:[")[1]).split("]")[0]
                tmp_lat_list = tmp_lat_str.split(",")
                true_final_coor_list["lng"] = tmp_lat_list
                print("true_final_lng:", true_final_coor_list["lng"])
            if not line:
                break
    with open(final_area_file_dir + "predicted/"+ordered_by_time_or_code+"_order"+"/FinalCarInArea.txt", "r") as pre_final_area_file:

        while True:
            line = pre_final_area_file.readline()
            if line.startswith("lat list"):
                tmp_lat_str = (line.split("lat list:[")[1]).split("]")[0]
                tmp_lat_list = tmp_lat_str.split(", ")
                pre_final_coor_list["lat"] = tmp_lat_list
                print("pre_final_lat:", pre_final_coor_list["lat"])
                pre_list_len += 1
            elif line.startswith("lng list"):
                tmp_lng_str = (line.split("lng list:[")[1]).split("]")[0]
                tmp_lng_list = tmp_lng_str.split(",")
                pre_final_coor_list["lng"] = tmp_lng_list
                print("pre_final_lng:", pre_final_coor_list["lng"])
            if not line:
                break



    print("pre list:",len(pre_final_coor_list["lat"]))
    print("true list:", len(true_final_coor_list["lat"]))
    pre_list_len = len(pre_final_coor_list["lat"])
    true_list_len = len(true_final_coor_list["lat"])
    # if len(pre_final_coor_list["lat"]) == len(true_final_coor_list["lat"]):
    # if pre_list_len == true_list_len:

    if len(pre_final_coor_list["lat"]) < len(true_final_coor_list["lat"]):
        diff = true_list_len - pre_list_len
        for i in range(diff):
            pre_final_coor_list["lat"].append(0)
            pre_final_coor_list["lng"].append(0)
    elif len(pre_final_coor_list["lat"]) > len(true_final_coor_list["lat"]):
        diff = pre_list_len - true_list_len
        for i in range(diff):
            true_final_coor_list["lat"].append(0)
            true_final_coor_list["lng"].append(0)
    for i in range(len(pre_final_coor_list["lat"])):
        # true - pre
        lat_offset_list.append(float(pre_final_coor_list["lat"][i]) - float(true_final_coor_list["lat"][i]))
        lng_offset_list.append(float(pre_final_coor_list["lng"][i]) - float(true_final_coor_list["lng"][i]))

    # lat_offset_sum = sum(abs(x) for x in lat_offset_list)
    lat_offset_sum = sum(x for x in lat_offset_list)
    lng_offset_sum = sum(x for x in lng_offset_list)
    print("pre_final_lat:", pre_final_coor_list["lat"])
    print("pre_final_lng:", pre_final_coor_list["lng"])
    print("true_final_lat:", true_final_coor_list["lat"])
    print("true_final_lng:", true_final_coor_list["lng"])

    performance_evaluation_file = open(final_area_file_dir + "predicted/"+ordered_by_time_or_code+"_order/"+"evaluation.txt","w")
    performance_evaluation_file.write("pre_final_lat:"+str(pre_final_coor_list["lat"])+"\n")
    performance_evaluation_file.write("pre_final_lng:"+str(pre_final_coor_list["lng"])+"\n")
    performance_evaluation_file.write("true_final_lat:"+str(true_final_coor_list["lat"])+"\n")
    performance_evaluation_file.write("true_final_lng:"+str(true_final_coor_list["lng"])+"\n")
    performance_evaluation_file.write("lat offset list:"+str(lat_offset_list)+"\n")
    performance_evaluation_file.write("lat offset sum:"+str(lat_offset_sum)+"\n")
    performance_evaluation_file.write("lng offset list:"+str(lng_offset_list)+"\n")
    performance_evaluation_file.write("lng offset sum:"+str(lng_offset_sum))


if __name__ == '__main__':


    carNum = '500'
    processingData_root += "taxi_"+carNum+"/"
    ordered_by_time_or_code = "code"
    if ordered_by_time_or_code == "code":
        showTaxiDataByCarCode(22575)
    else:
        showTaxisDataBetweenTime(1,2)

    performance_evaluation(processingData_root, ordered_by_time_or_code)
    #
    # # ordered_file_directory = "./processingData/rearrangeTaxiTime/taxiCode/sortByTimeInSequence/"
    # # filename = "taxiCode_22223"
    # carNum = '100'
    # car_code = str(22523)
    # true_or_pred = "true"
    # ordered_file_directory = "./processingData/taxi_"+carNum+'/'+true_or_pred+'/order/'
    #
    # pred_filename = "predicted_trajectory"
    # true_filename = "true_trajectory"
    # file_path = "savedata/taxi_" + str(carNum) + "/GATraj/"
    #
    # mapHtmlpath = "result_data/"+ true_or_pred+"_"+ car_code+ ".html"
    # regionHtmlPath = "result_data/"+ true_or_pred+"_region_"+ car_code+ ".html"
    # trajectoryPath = "result_data/"+ true_or_pred+"_trajectory_"+car_code + ".html"
    # taxiData, code_time_list = readLatAndLngFromFile(ordered_file_directory+'taxiCode_'+str(car_code)+'.txt') #读取经纬度
    #
    # map = drawscattByPartition(taxiData, mapHtmlpath) # 画出车辆经纬度图
    # # region = dynamicAreaParition(taxiData)     # 划分区域
    # # drawCarDensity(region, map, regionHtmlPath) # 画出车辆密度图
    # drawCartrajectory(taxiData, code_time_list, map, trajectoryPath) # 画出车辆轨迹
    # outputDisToFile(taxiData, code_time_list, True)  # 输出两点间距离,输出到processingData目录下
    # # print("The avg time interval is ", calAvgTimeInterval(code_time_list)[0],"s") # 计算平均时间
    #
    # # sortAreaByCarNum("./processingData/FinalCarInArea.txt", "./processingData/FinalCarInAreaWithoutZero.txt")
    #
    #
    #
