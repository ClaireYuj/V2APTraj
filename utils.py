import torch
import os
import pickle
import numpy as np
import random
from tqdm import tqdm


class DataLoader_bytrajec2():
    def __init__(self, args, is_gt=True):
        self.miss = 0
        self.args = args
        self.is_gt = is_gt
        self.num_tra = 0
        if self.args.dataset == 'eth5':

            self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                              './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                              './data/ucy/univ/students001', 'data/ucy/univ/students003',
                              './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']
            self.data_dir = './data'
            # self.data_dirs = ['./sample/test', './sample/22231_22235']
            # self.data_dir = './'
            # Data directory where the pre-processed pickle file resides

            # skip=[6,10,10,10,10,10,10,10]
            skip = [6, 10, 10, 10, 10, 10, 10, 10]
            if args.ifvalid:
                self.val_fraction = args.val_fraction
            else:
                self.val_fraction = 0

            train_set = [i for i in range(len(self.data_dirs))]
            if args.test_set == 4 or args.test_set == 5:
                self.test_set = [4, 5]
            else:
                self.test_set = [self.args.test_set]

            for x in self.test_set:
                train_set.remove(x)
            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]

            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]

        if self.args.dataset == 'taxi':
            # self.data_dirs = ['./sample_lnglat/test', './sample_lnglat/22223_22523','./sample_lnglat/22333_22343']
            self.data_dirs = ['./sample_lnglat/test', './sample_lnglat/22223_22523']
            self.data_dir = './'
            # Data directory where the pre-processed pickle file resides

            # skip=[6,10,10,10,10,10,10,10]
            skip = [10, 10, 10, 10, 10, 10, 10, 10]
            if args.ifvalid:
                self.val_fraction = args.val_fraction
            else:
                self.val_fraction = 0

            train_set = [i for i in range(len(self.data_dirs))]
            # if args.test_set==4 or args.test_set==5:
            #     self.test_set=[4,5]
            # else:
            #     self.test_set=[self.args.test_set]
            self.test_set = [self.args.test_set]
            for x in self.test_set:
                train_set.remove(x)
            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            print("train dir:", self.train_dir)
            print("test dir:", self.test_dir)
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]

        self.train_data_file = os.path.join(self.args.save_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.save_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.save_dir, "train_batch_cache.cpkl")
        self.test_batch_cache = os.path.join(self.args.save_dir, "test_batch_cache.cpkl")

        print("Creating pre-processed data from raw data.")
        self.traject_preprocess('train')
        self.traject_preprocess('test')
        print("Done.")

        # Load the processed data from the pickle file
        print("Preparing data batches.")
        if not (os.path.exists(self.train_batch_cache)):
            self.frameped_dict, self.pedtraject_dict = self.load_dict(self.train_data_file)
            self.dataPreprocess('train')
            print("self.num_tra", self.num_tra)
            self.num_tra = 0
        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict = self.load_dict(self.test_data_file)
            self.dataPreprocess('test')
            print("self.num_tra", self.num_tra)
        self.trainbatch, self.trainbatchnums, \
            self.valbatch, self.valbatchnums = self.load_cache(self.train_batch_cache)
        self.testbatch, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache)
        print("Done.")

        print('Total number of training batches:', self.trainbatchnums)
        # print('self.trainbatch:', self.trainbatch)
        print('Total number of validation batches:', self.valbatchnums)
        print('Total number of test batches:', self.testbatchnums)

    def traject_preprocess(self, setname):
        '''
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:
            data_dirs = self.test_dir
            data_file = self.test_data_file
        all_frame_data = []
        valid_frame_data = []
        numFrame_data = []

        Pedlist_data = []  # 车辆id list
        frameped_dict = []  # peds id contained in a certain frame b
        pedtrajec_dict = []  # trajectories of a certain ped
        # For each dataset
        for seti, directory in enumerate(data_dirs):

            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset
            Pedlist = np.unique(data[1, :]).tolist()  # 车辆id list
            numPeds = len(Pedlist)
            # Add the list of frameIDs to the frameList_data
            Pedlist_data.append(Pedlist)
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])
            numFrame_data.append([])
            frameped_dict.append({})
            pedtrajec_dict.append({})

            for ind, pedi in enumerate(Pedlist):
                # if ind%100==0:
                #     print(ind,len(Pedlist))
                # Extract trajectories of one person
                FrameContainPed = data[:, data[1, :] == pedi]  # 获取每一列（也就是每一个id的车的时间，id，经纬度）
                # Extract peds list
                FrameList = FrameContainPed[0, :].tolist()  # 获取当前id车辆的frame时间
                if len(FrameList) < 2:  # 剔除那些只有一个数据的
                    continue
                # Add number of frames of this trajectory
                numFrame_data[seti].append(len(FrameList))  # 记录当前车id出现的帧数有多少
                # Initialize the row of the numpy array
                Trajectories = []
                # For each ped in the current frame
                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                    current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                    # Add their pedID, x, y to the row of the numpy array
                    Trajectories.append([int(frame), current_x, current_y])
                    if int(frame) not in frameped_dict[seti]:
                        frameped_dict[seti][int(frame)] = []  # 以帧数为key，car_id为value创建dic
                    frameped_dict[seti][int(frame)].append(pedi)
                pedtrajec_dict[seti][pedi] = np.array(Trajectories)

        with open(data_file, "wb") as f:  # frameded_dict -- 每一帧有谁的cic
            pickle.dump((frameped_dict, pedtrajec_dict), f, protocol=2)

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]
        return frameped_dict, pedtraject_dict

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname == 'train':
            val_fraction = self.args.val_fraction
            frameped_dict = self.frameped_dict
            pedtraject_dict = self.pedtraject_dict
            cachefile = self.train_batch_cache

        else:
            val_fraction = 0
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            cachefile = self.test_batch_cache

        data_index = self.get_data_index(frameped_dict, setname)  # to split data in different batch

        val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]

        # trainbatch=self.get_seq_from_index_balance(frameped_dict,pedtraject_dict,train_index,setname)
        # valbatch = self.get_seq_from_index_balance(frameped_dict,pedtraject_dict,val_index,setname)
        trainbatch = self.get_seq_from_index_balance_with_ID(frameped_dict, pedtraject_dict, train_index, setname)
        valbatch = self.get_seq_from_index_balance_with_ID(frameped_dict, pedtraject_dict, val_index, setname)

        trainbatchnums = len(trainbatch)
        valbatchnums = len(valbatch)
        # print("trainbatch:{0},\n trainbatchnums:{1},\n valbatch:{2},\n valbatchnums:{3}\n".format(trainbatch, trainbatchnums, valbatch, valbatchnums))
        f = open(cachefile, "wb")
        pickle.dump((trainbatch, trainbatchnums, valbatch, valbatchnums), f, protocol=2)
        f.close()

    def get_data_index(self, data_dict, setname, ifshuffle=True):
        '''
        Get the dataset sampling index.
        set用于标识场景
        '''
        # print("data dict:",data_dict)
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        for seti, dict in enumerate(data_dict):  # 此处的data dic是以帧为key, car_id为 value的dic
            frames = sorted(dict)  # frames is the list sorted by time asc
            maxframe = max(frames) - self.args.pred_length
            frames = [x for x in frames if not x > maxframe]
            total_frame += len(frames)
            set_id.extend(list(seti for i in range(len(frames))))
            frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))
        all_frame_id_list = list(i for i in range(total_frame))

        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)),
                                    0)  # data_index是一个二维list, data_list = [[frame time], [出现的第几次], [frame_id]
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]  # 将data_index以每列为单元打乱

        # to make full use of the data
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)

        return data_index  # 返回一个[[帧], [场景id], [帧id]]的list,场景也就是训练集

    def get_seq_from_index_balance_with_ID(self, frameped_dict, pedtraject_dict, data_index, setname):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''
        batch_data_mass = []
        batch_data = []
        Batch_id = []
        ped_list = []  # 用于存储对应的行人标识信息
        time_seq_list = ()  # 用于存储时间

        if setname == 'train':
            skip = self.trainskip
        else:
            skip = self.testskip

        ped_cnt = 0
        batch_count = 0
        batch_data_64 = []
        batch_split = []
        start, end = 0, 0
        nei_lists = []
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, '/number of frames of data in total', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            try:
                frameend_pedi = set(
                    frameped_dict[cur_set][cur_frame + (self.args.pred_length - 1 + self.args.min_obs) * skip[cur_set]])
                # 设置frame结束点
                # frameend_pedi = set(frameped_dict[cur_set][cur_frame+300])
            except:
                if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
                    batch_data_mass.append((ped_list, batch_data_64, time_seq_list, batch_split, nei_lists,))
                continue
            present_pedi = framestart_pedi | frameend_pedi  # 取两个集合的并集，表示在当前帧开始和结束时刻存在的行人
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
                    batch_data_mass.append((ped_list, batch_data_64, time_seq_list, batch_split, nei_lists,))
                continue
            traject = ()  # 20个一组，32组
            for ped in present_pedi:
                cur_trajec, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped], cur_frame,
                                                                       self.args.seq_length, skip[cur_set])
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    # Just ignore trajectories if their data don't exist at the last observed time step (easy for data shift)
                    continue
                # cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, self.args.input_size),)  # 不包含行人标识信息和时间信息
                cur_time = (cur_trajec[:, 0].reshape(-1, 1, 1),)
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, self.args.input_size),)

                traject = traject.__add__(cur_trajec)  # tuple of cur_trajec arrays in the same scene
                ped_list.append(ped)  # 将行人标识信息存储到 ped_list 中
                time_seq_list = time_seq_list.__add__(cur_time)
            if traject.__len__() < 1:
                if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
                    batch_data_mass.append((ped_list, batch_data_64, time_seq_list, batch_split, nei_lists,))
                continue
            self.num_tra += traject.__len__()
            end += traject.__len__()  # batch是按照 traject 的长度划分的
            batch_split.append([start, end])
            start = end
            traject_batch = np.concatenate(traject, 1)  # ped dimension
            cur_pednum = traject_batch.shape[1]  # 当前的 batch split size
            batch_id = (cur_set, cur_frame,)
            cur_batch_data, cur_Batch_id = [], []
            cur_batch_data.append(traject_batch)  # 向 cur_batch_data 中加入轨迹数据
            cur_Batch_id.append(batch_id)
            cur_batch_data, nei_list = self.massup_batch(cur_batch_data)
            nei_lists.append(nei_list)
            ped_cnt += cur_pednum
            batch_count += 1
            if self.args.batch_size == 1:
                batch_data_mass.append((ped_list, cur_batch_data, time_seq_list, batch_split, nei_lists,))
                batch_split = []
                start, end = 0, 0
                nei_lists = []
                ped_list = []  # 清空 ped_list
            else:
                if batch_count == self.args.batch_size or i == data_index.shape[1] - 1:
                    batch_data_64 = self.merg_batch(cur_batch_data, batch_data_64)
                    batch_data_mass.append((ped_list, batch_data_64, time_seq_list, batch_split, nei_lists,))
                    batch_count = 0
                    batch_split = []
                    start, end = 0, 0
                    nei_lists = []
                    ped_list = []  # 清空 ped_list
                else:
                    if batch_count == 1:
                        batch_data_64 = cur_batch_data
                    else:
                        batch_data_64 = self.merg_batch(cur_batch_data, batch_data_64)
        return batch_data_mass  # [[id], [lng,lat], [time], [batch_split],[nei_list]]

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''
        batch_data_mass = []
        batch_data = []
        Batch_id = []

        if setname == 'train':
            skip = self.trainskip
        else:
            skip = self.testskip

        ped_cnt = 0
        batch_count = 0
        batch_data_64 = []
        batch_split = []
        start, end = 0, 0
        nei_lists = []
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, '/number of frames of data in total', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            try:
                frameend_pedi = set(
                    frameped_dict[cur_set][cur_frame + (self.args.pred_length - 1 + self.args.min_obs) * skip[cur_set]])
                # 设置frame结束点
                # frameend_pedi = set(frameped_dict[cur_set][cur_frame+300])
            except:
                if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
                    batch_data_mass.append((batch_data_64, batch_split, nei_lists,))
                continue
            present_pedi = framestart_pedi | frameend_pedi  # 表示取两个集合的并集。具体来说，framestart_pedi 和 frameend_pedi 是两个集合，分别表示在当前帧开始和结束时刻存在的行人
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
                    batch_data_mass.append((batch_data_64, batch_split, nei_lists,))
                continue
            traject = ()  # 20个一组，32组
            for ped in present_pedi:
                cur_trajec, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped], cur_frame,
                                                                       self.args.seq_length, skip[cur_set])
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, self.args.input_size),)
                traject = traject.__add__(cur_trajec)  # tuple of cur_trajec arrays in the same scene
            if traject.__len__() < 1:
                if i == data_index.shape[1] - 1 and self.args.batch_size != 1:
                    batch_data_mass.append((batch_data_64, batch_split, nei_lists,))
                continue
            self.num_tra += traject.__len__()
            end += traject.__len__()  # batch是按照traject的长度划分的
            batch_split.append([start, end])
            start = end
            traject_batch = np.concatenate(traject, 1)  # ped dimension
            cur_pednum = traject_batch.shape[1]  # 当前的batch split size
            batch_id = (cur_set, cur_frame,)
            cur_batch_data, cur_Batch_id = [], []
            cur_batch_data.append(traject_batch)  # 向cur_batch_data中加入轨迹数据
            cur_Batch_id.append(batch_id)
            cur_batch_data, nei_list = self.massup_batch(cur_batch_data)
            nei_lists.append(nei_list)
            ped_cnt += cur_pednum
            batch_count += 1
            if self.args.batch_size == 1:
                batch_data_mass.append((cur_batch_data, batch_split, nei_lists,))
                batch_split = []
                start, end = 0, 0
                nei_lists = []
            else:
                if batch_count == self.args.batch_size or i == data_index.shape[1] - 1:
                    batch_data_64 = self.merg_batch(cur_batch_data, batch_data_64)
                    batch_data_mass.append((batch_data_64, batch_split, nei_lists,))
                    batch_count = 0
                    batch_split = []
                    start, end = 0, 0
                    nei_lists = []
                else:
                    if batch_count == 1:
                        batch_data_64 = cur_batch_data
                    else:
                        batch_data_64 = self.merg_batch(cur_batch_data, batch_data_64)
        return batch_data_mass

    def merg_batch(self, cur_batch_data, batch_data_64):
        merge_batch_data = []
        for cur_data, data_64 in zip(cur_batch_data, batch_data_64):
            merge = np.concatenate([data_64, cur_data], axis=0)
            merge_batch_data.append(merge)

        return merge_batch_data

    def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip):
        '''
        Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
        '''
        return_trajec = np.zeros((seq_length, self.args.input_size + 1))
        endframe = startframe + (self.args.pred_length - 1 + self.args.min_obs) * skip
        start_n = np.where(trajectory[:, 0] == startframe)
        end_n = np.where(trajectory[:, 0] == endframe)
        ifexsitobs = False
        real_startframe = startframe
        offset_start = self.args.obs_length - self.args.min_obs
        if start_n[0].shape[0] != 0 and end_n[0].shape[0] != 0:
            end_n = end_n[0][0]
            for i in range(0, self.args.obs_length - self.args.min_obs + 1):
                if np.where(trajectory[:, 0] == startframe - (self.args.obs_length - self.args.min_obs - i) * skip)[
                    0].shape[0] != 0:
                    real_startframe = startframe - (self.args.obs_length - self.args.min_obs - i) * skip
                    start_n = np.where(trajectory[:, 0] == real_startframe)[0][0]
                    offset_start = i
                    break
        else:
            return return_trajec, ifexsitobs

        candidate_seq = trajectory[start_n:end_n + 1]
        try:
            return_trajec[offset_start:, :] = candidate_seq
            if offset_start > 0:
                return_trajec[:offset_start, :] = candidate_seq[0, :]
        except:
            self.miss += 1
            return return_trajec, ifexsitobs

        if return_trajec[self.args.obs_length - 1, 1] != 0:
            ifexsitobs = True

        return return_trajec, ifexsitobs

    def massup_batch(self, batch_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]
        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, self.args.input_size))
        nei_list_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        nei_num_b = np.zeros((self.args.seq_length, num_Peds))
        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            num_Ped = batch.shape[1]
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch)
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
            batch_data = (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
        return self.get_dm_offset(batch_data)

    def get_dm_offset(self, inputs):
        """   batch_abs: the (orientated) batch [H, N, inputsize] inputsize: x,y,z,yaw,h,w,l,label
        batch_norm: the batch shifted by substracted the last position.
        shift_value: the last observed position
        seq_list: [seq_length, num_peds], mask for position with actual values at each frame for each ped
        nei_list: [seq_length, num_peds, num_peds], mask for neigbors at each frame
        nei_num: [seq_length, num_peds], neighbors at each frame for each ped
        batch_pednum: list, number of peds in each batch"""
        nodes_abs, seq_list, nei_list, nei_num, batch_pednum = inputs
        cur_ori = nodes_abs.copy()
        cur_ori, seq_list = cur_ori.transpose(1, 0, 2), seq_list.transpose(1, 0)
        nei_num = nei_num.transpose(1, 0)
        return [cur_ori, seq_list, nei_num], nei_list  # [N, H], [N, H], [N, H], [H, N, N

    def get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing
        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1
        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                seqj = inputnodes[:, pedj]
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)
                relative_cord = seqi[select, :2] - seqj[select, :2]
                # invalid data index
                select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                            abs(relative_cord[:, 1]) > self.args.neighbor_thred)
                nei_num[select, pedi] -= select_dist
                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num

    def rotate_shift_batch(self, batch_data, epoch, idx, ifrotate=True):
        '''
        Random ration and zero shifting.
        Random rotation is also helpful for reducing overfitting.
        For one mini-batch, random rotation is employed for data augmentation.
        #[N, H, 2] [N, H], [N, G, G, 4] , (B, H, W) #[position, angle, framenum, ego or nei]
        '''
        nodes_abs, seq_list, nei_num = batch_data
        nodes_abs = nodes_abs.transpose(1, 0, 2)  # [H, N, 2]
        # rotate batch
        if ifrotate:
            th = np.random.random() * np.pi
            cur_ori = nodes_abs.copy()
            nodes_abs[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            nodes_abs[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)
        s = nodes_abs[self.args.obs_length - 1, :, :2]
        # ，shift the origin to the latest observed time step
        shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)
        batch_data = nodes_abs, nodes_abs[:, :, :2] - shift_value, shift_value, seq_list, nei_num
        return batch_data

    def get_train_batch(self, idx, epoch):
        id_lists, batch_data, time_seq_list, batch_split, nei_lists = self.trainbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, epoch, idx, ifrotate=self.args.randomRotate)

        return id_lists, batch_data, time_seq_list, batch_split, nei_lists

    def get_val_batch(self, idx, epoch):
        id_lists, batch_data, time_seq_list, batch_split, nei_lists = self.valbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, epoch, idx, ifrotate=False)
        return id_lists, batch_data, time_seq_list, batch_split, nei_lists

    def get_test_batch(self, idx, epoch):
        id_lists, batch_data, time_seq_list, batch_split, nei_lists = self.testbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, epoch, idx, ifrotate=False)
        return id_lists, batch_data, time_seq_list, batch_split, nei_lists


def getLossMask(outputs, node_first, seq_list, using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exsist.
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exsist.
    '''
    seq_length = outputs.shape[0]
    node_pre = node_first
    lossmask = torch.zeros(seq_length, seq_list.shape[1])
    if using_cuda:
        lossmask = lossmask.cuda()
    for framenum in range(seq_length):
        lossmask[framenum] = seq_list[framenum] * node_pre
        if framenum > 0:
            node_pre = seq_list[framenum - 1]
    return lossmask, sum(sum(lossmask))


def L2forTest(outputs, targets, obs_length):
    '''
    Evaluation.
    information: [N, 3]
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)

    error_pred_length = error[obs_length - 1:]
    error = torch.sum(error_pred_length)
    error_cnt = error_pred_length.numel()
    if error == 0:
        return 0, 0, 0, 0, 0, 0
    final_error = torch.sum(error_pred_length[-1])
    final_error_cnt = error_pred_length[-1].numel()
    first_erro = torch.sum(error_pred_length[0])
    first_erro_cnt = error_pred_length[0].numel()
    error = torch.mean(error ** 2)
    return error.item(), error_cnt, final_error.item(), final_error_cnt, first_erro.item(), first_erro_cnt


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod