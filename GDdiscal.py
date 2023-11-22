import math
import numpy as np
import folium
import webbrowser
from folium import plugins
import pandas as pd
from folium.plugins import HeatMap
import time
import pyproj

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



def outputDisToFile(taxiData, code_time_data, isOrderByTime):
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


def areaParition(datalist):
    """
    划分区域,
    传入的是经纬度的list data= [[lat1, lng1], [lat2, lng2]]
    可以利用全局变量 SUB_AREA_LENGTH 和 SUB_AREA_WIDTH控制区域大小
    返回画好的区域的字典数组，可以在FinialCarInArea.txt文档中查看
    :param datalist: 车辆经纬度的列表[[laat1,lng1],[lat2,lng2]
    :return:
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
    test_area_file = open(processingData_root+"testAreaPart.txt", "w")
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

    car_record_file = open(processingData_root+"CarRecord.txt", "w")
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
    final_area_recode_file = open(processingData_root+"FinialCarInArea.txt", "w")
    for m in region_map:
        if m["carNum"] > 0:
            final_area_recode_file.write(str(m)+"\n")
    print("finish write the car in file......")
    final_area_recode_file.close()
    car_record_file.close()
    test_area_file.close()
    return region_map



def drawCarDensity(region_map, schools_map, regionHtmlPath):
    """
    在地图上标出区域的范围， 并显示该区域内的车辆数
    :param region_map:  areaParition的返回值，形如[{"area index": 1, "min_lat":10, "min_lng":10, "max_lat":20, "max_lng":20, "carNum":30},{...},{...}]
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
            folium.Rectangle(
                bound,
                # popup="</br>car num:"+str(region["carNum"])+"</br>",
                tooltip="</br>car num:"+str(region["carNum"])+"</br>",
                fill=True
            ).add_to(schools_map)
            # print(" draw one : min_lat:", min_lat, "min_lng:", min_lng, "max_lat:", max_lat, "max_lng:", max_lng)

    schools_map.save(regionHtmlPath)


def drawCartrajectory(taxiData, code_time_list, map ,new_map_path):
    """
    画出行车轨迹
    :param taxiData: 包含经纬度的列表，形如[[lat1, lng1], [lat2,lng2]
    :param code_time_list: 包含车辆编码和时间的列表，如[[22223,00:00:01],[22223,10:02:01]
    :param map: 已经标号点的地图，foilum.Map变量，（是drawscatter或者drawscatterByPartition的返回值）
    :param new_map_path: 存放行车轨迹的路径
    :return:
    """
    print("Start draw trajectory.....")
    locations = [taxiData[0]]
    lenOfData = len(taxiData)
    for i in range(1, lenOfData):
        point = taxiData[i]
        locations.append(point)
        data_0 = code_time_list[i-1]
        data_1 = code_time_list[i]
        taxi_info = "code: "+str(data_0[0])+" time: "+str(data_0[1])+"-"+str(data_1[1])
        # 包含车辆信息
        polyline = folium.PolyLine(locations=locations, color="red", weight=2.5, opacity=0.5,
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
    map.save(new_map_path)





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
def drawscatt(data, mapHtmlpath):
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


def drawscattByPartition(taxiData, mapHtmlpath):
    """
    难以直接把所有的点一次性画出来， 所以分步画
    :param taxiData: 包含经纬度的列表，形如[[lat1, lng1], [lat2, lng2]
    :param mapHtmlpath: 输出html地址
    :return:
    """
    print("start drawing partition.......")
    lenOfData = len(taxiData)
    size = 100000
    count = 0

    schools_map = folium.Map(location=taxiData[0], zoom_start=10,
                             tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}",
                             attr="&copy; <a href='http://ditu.amap.com/'>高德地图</a>"
                             )
    marker_cluster = plugins.MarkerCluster().add_to(schools_map)
    start_time = time.time()
    index = 0
    while index > lenOfData:
        try:

            subData = taxiData[index:index+size]

            for j in range(len(subData)):
                string_car = str(subData[j])
                info = "code: "+string_car.split(",")[0]+" time: "+string_car.split(",")[1]
                folium.Marker(subData[j]).add_to(
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

            count += 1
            try:
                # lat = float(line.split(",")[3])
                # lng = float(line.split(",")[2])
                # code = int(line.split(",")[0])
                # time = str(line.split(",")[1])
                lat = float(line.split(",")[4])
                lng = float(line.split(",")[3])
                code = int(line.split(",")[1])
                time = str(line.split(",")[2])

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
        if line:
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


# 计算gcj中两点的距离
# 将点显示在高德地图中，已经验证符合实际位置，可以使用
if __name__ == '__main__':


    # taxiData = [[22.560682, 114.157585]]
    # mapHtmlpath=r"F:\pyworkspace\casesData\src\utils\img\高德地图数据测试.html"

    # ordered_file_directory = "./processingData/rearrangeTaxiTime/taxiCode/sortByTimeInSequence/"
    # filename = "taxiCode_22223"
    ordered_file_directory = "./processingData/filterArea/order/"
    filename = "taxiCode_22224"

    mapHtmlpath = r"./casesData/src/utils/img/高德地图数据测试_" +"已排序_"+ "train_" + filename + ".html"
    regionHtmlPath = r"./casesData/src/utils/img/高德地图数据测试_区域划分_"+ "已排序_"+ "train_" + filename + ".html"
    trajectoryPath = r"./casesData/src/utils/img/高德地图数据测试_行车轨迹_"+ "已排序_"+ "train_" + filename + ".html"

    taxiData, code_time_list = readLatAndLngFromFile(ordered_file_directory + filename + ".txt") #读取经纬度

    map = drawscattByPartition(taxiData, mapHtmlpath) # 画出车辆经纬度图
    region = areaParition(taxiData)     # 划分区域
    drawCarDensity(region, map, regionHtmlPath) # 画出车辆密度图
    drawCartrajectory(taxiData, code_time_list, map, trajectoryPath) # 画出车辆轨迹
    outputDisToFile(taxiData, code_time_list, True)  # 输出两点间距离,输出到processingData目录下
    print("The avg time interval is ", calAvgTimeInterval(code_time_list)[0],"s") # 计算平均时间

    # sortAreaByCarNum("./processingData/FinialCarInArea.txt", "./processingData/FinalCarInAreaWithoutZero.txt")



