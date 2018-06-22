
import numpy as np
import os
import random

from tools.gaze_data import GazeData


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





class ColumbiaGaze:
    def __init__(self):
        self.min_num_examples = int(10000)
        self.gaze_data = [] # pose,person
        self.num_persons = None
        self.num_pose = None
        self.folder_list = ['H-30', 'H-15', 'H0', 'H15', 'H30']

    def load_columbia_gaze(self, is_left, data_path):
        assert len(self.gaze_data) == 0

        if os.path.isdir(data_path) == False:
            raise ValueError('no such data path {}'.format(data_path))

        for dir in self.folder_list:
            one_pose_full_path = os.path.join(data_path, dir)
            if os.path.isdir(one_pose_full_path) == True:
                self.num_pose = 1 if self.num_pose is None else self.num_pose + 1
                self.load_by_pose(is_left, one_pose_full_path)

        # output the data information
        print('columbia gaze data: total {}..., max batch size {}'.format(len(self.gaze_data), self.min_num_examples))

    def load_by_pose(self, is_left, data_path):
        person_counter = 0
        dir_list = os.listdir(data_path)
        for dir in dir_list:
            if '.mat' in dir:
                full_path = os.path.join(data_path, dir)
                self.load_one_mat(is_left, full_path)
                # self.load_one_mat(True, full_path)
                # self.load_one_mat(False, full_path)
                person_counter += 1
            else:
                # skip the file
                pass
        # min number of persons of each pose
        if self.num_persons is None:
            self.num_persons = person_counter
        else:
            self.num_persons = self.num_persons if self.num_persons < person_counter else person_counter

    def load_one_mat(self, is_left, full_path):
        data = GazeData()
        data.load_columbia_gaze(is_left=is_left, data_path=full_path)
        # data.load_self_gaze(is_left=is_left, data_path=full_path)

        # append to list and update max batch size
        self.gaze_data.append(data)
        # self.num_examples.append(data.num_examples)
        self.min_num_examples = data.num_examples if data.num_examples < self.min_num_examples else self.min_num_examples
        if data.num_examples == 4:
            print(full_path)

    def next_one_batch_pair(self, index=None, shuffle=True):
        batch_size = 10
        pair1_eye_batch_list = []
        pair2_eye_batch_list = []
        pair1_angle_batch_list = []
        pair2_angle_batch_list = []
        pair1_coord_batch_list = []
        pair2_coord_batch_list = []

        # the 1st part
        pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_batch(batch_size, shuffle)
        pair1_eye_batch_list.append(pair_eye)
        pair1_angle_batch_list.append(pair_angle)
        pair1_coord_batch_list.append(pair_coord)
        # the 2nd part
        pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_batch(batch_size, shuffle)
        pair2_eye_batch_list.append(pair_eye)
        pair2_angle_batch_list.append(pair_angle)
        pair2_coord_batch_list.append(pair_coord)

        # concatenate together and return
        pair1_eye = np.concatenate(pair1_eye_batch_list, axis=0)
        pair2_eye = np.concatenate(pair2_eye_batch_list, axis=0)
        pair1_angle = np.concatenate(pair1_angle_batch_list, axis=0)
        pair2_angle = np.concatenate(pair2_angle_batch_list, axis=0)
        pair1_coord = np.concatenate(pair1_coord_batch_list, axis=0)
        pair2_coord = np.concatenate(pair2_coord_batch_list, axis=0)
        return pair1_eye, pair2_eye, pair1_angle, pair2_angle, pair1_coord, pair2_coord

    def next_batch_pair(self, batch_size, shuffle=True):
        # use all pose and all person to train
        # at least 1 examples
        batch_size_one_person = self.min_num_examples // 2
        assert batch_size_one_person > 0
        assert batch_size > batch_size_one_person
        assert batch_size_one_person * len(self.gaze_data) > batch_size

        pair1_eye_batch_list = []
        pair2_eye_batch_list = []
        pair1_angle_batch_list = []
        pair2_angle_batch_list = []
        pair1_coord_batch_list = []
        pair2_coord_batch_list = []
        sum_batch_size = 0

        # random select some persons
        num_select_person = batch_size // batch_size_one_person + 1
        idx_person = random.sample(range(len(self.gaze_data)), num_select_person)
        for index in idx_person:
            # calculate the batch size for each person
            if (sum_batch_size+batch_size_one_person) < batch_size:
                cur_batch_size = batch_size_one_person
            else:
                cur_batch_size = batch_size - sum_batch_size

            # the 1st part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_batch(cur_batch_size, shuffle)
            pair1_eye_batch_list.append(pair_eye)
            pair1_angle_batch_list.append(pair_angle)
            pair1_coord_batch_list.append(pair_coord)
            # the 2nd part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_batch(cur_batch_size, shuffle)
            pair2_eye_batch_list.append(pair_eye)
            pair2_angle_batch_list.append(pair_angle)
            pair2_coord_batch_list.append(pair_coord)
            # add to
            sum_batch_size += cur_batch_size

        # concatenate together and return
        pair1_eye = np.concatenate(pair1_eye_batch_list, axis=0)
        pair2_eye = np.concatenate(pair2_eye_batch_list, axis=0)
        pair1_angle = np.concatenate(pair1_angle_batch_list, axis=0)
        pair2_angle = np.concatenate(pair2_angle_batch_list, axis=0)
        pair1_coord = np.concatenate(pair1_coord_batch_list, axis=0)
        pair2_coord = np.concatenate(pair2_coord_batch_list, axis=0)
        return pair1_eye, pair2_eye, pair1_angle, pair2_angle, pair1_coord, pair2_coord

    def next_group_batch_pair(self, batch_size, reduce_ratio=2, inside_shuffle=False):
        # use all pose and all person to train
        # at least 1 examples
        batch_size_one_person = self.min_num_examples // (GazeData.num_groups*reduce_ratio)
        assert batch_size_one_person > 0
        # assert batch_size > batch_size_one_person
        assert batch_size_one_person * len(self.gaze_data) > batch_size

        pair1_eye_batch_list = []
        pair2_eye_batch_list = []
        pair1_angle_batch_list = []
        pair2_angle_batch_list = []
        pair1_coord_batch_list = []
        pair2_coord_batch_list = []
        sum_batch_size = 0

        # random select some persons
        num_select_person = batch_size // batch_size_one_person + 1
        idx_person = random.sample(range(len(self.gaze_data)), num_select_person)

        for index in idx_person:
            # calculate the batch size for each person
            if (sum_batch_size+batch_size_one_person) < batch_size:
                cur_batch_size = batch_size_one_person
            else:
                cur_batch_size = batch_size - sum_batch_size

            # generate group index, we only need two
            group_idx = random.sample(range(GazeData.num_groups), 2)

            # the 1st part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_group_batch(cur_batch_size, group_idx[0], inside_shuffle)
            pair1_eye_batch_list.append(pair_eye)
            pair1_angle_batch_list.append(pair_angle)
            pair1_coord_batch_list.append(pair_coord)
            # the 2nd part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_group_batch(cur_batch_size, group_idx[1], inside_shuffle)
            pair2_eye_batch_list.append(pair_eye)
            pair2_angle_batch_list.append(pair_angle)
            pair2_coord_batch_list.append(pair_coord)
            # add to
            sum_batch_size += cur_batch_size

        # concatenate together and return
        pair1_eye = np.concatenate(pair1_eye_batch_list, axis=0)
        pair2_eye = np.concatenate(pair2_eye_batch_list, axis=0)
        pair1_angle = np.concatenate(pair1_angle_batch_list, axis=0)
        pair2_angle = np.concatenate(pair2_angle_batch_list, axis=0)
        pair1_coord = np.concatenate(pair1_coord_batch_list, axis=0)
        pair2_coord = np.concatenate(pair2_coord_batch_list, axis=0)
        return pair1_eye, pair2_eye, pair1_angle, pair2_angle, pair1_coord, pair2_coord