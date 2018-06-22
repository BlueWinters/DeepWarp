
import numpy as np
import os
import random

from tools.gaze_data import GazeData


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





class SynthesisGaze:
    def __init__(self):
        self.min_num_examples = int(10000)
        self.gaze_data = [] # pose, person
        self.num_persons = None
        self.num_pose = None

    def load_synthesis_gaze(self, data_path):
        assert len(self.gaze_data) == 0
        if os.path.isdir(data_path) == False:
            raise ValueError('no such data path {}'.format(data_path))

        mat_counter = 0
        dir_list = os.listdir(data_path)
        for dir in dir_list:
            if '.mat' in dir:
                full_path = os.path.join(data_path, dir)
                self.load_one_mat(full_path)
                mat_counter += 1

        # output the data information
        print('columbia gaze data: total {}..., max batch size {}'.format(len(self.gaze_data), self.min_num_examples))

    def load_one_mat(self, full_path):
        data = GazeData()
        GazeData.num_groups = 15*2 # * images for one groups 15*20
        data.load_synthesis_gaze(data_path=full_path)

        # append to list and update max batch size
        self.gaze_data.append(data)
        self.min_num_examples = data.num_examples if data.num_examples < self.min_num_examples else self.min_num_examples

    def next_group_batch_pair(self, batch_size, inside_shuffle=True):
        pair1_eye_batch_list = []
        pair2_eye_batch_list = []
        pair1_angle_batch_list = []
        pair2_angle_batch_list = []
        pair1_coord_batch_list = []
        pair2_coord_batch_list = []
        sum_batch_size = 0

        # random select some persons
        num_select_person = len(self.gaze_data) // 3
        batch_size_one_person = batch_size // num_select_person
        assert batch_size_one_person > 3

        idx_person = random.sample(range(len(self.gaze_data)), num_select_person)

        for index in idx_person:
            # calculate the batch size for each person
            if (sum_batch_size+batch_size_one_person) < batch_size:
                cur_batch_size = batch_size_one_person
            else:
                cur_batch_size = batch_size - sum_batch_size

            easy_batch_size = cur_batch_size // 3 * 2
            hard_batch_size = cur_batch_size - easy_batch_size
            assert easy_batch_size > 0 and hard_batch_size > 0

            # easy & hard
            group_idx = random.sample(range(GazeData.num_groups/2), 3)

            # easy for the 1st part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_group_batch(easy_batch_size, group_idx[0] * 2, inside_shuffle)
            pair1_eye_batch_list.append(pair_eye)
            pair1_angle_batch_list.append(pair_angle)
            pair1_coord_batch_list.append(pair_coord)
            # easy for the 2nd part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_group_batch(easy_batch_size, group_idx[0] * 2 + 1, inside_shuffle)
            pair2_eye_batch_list.append(pair_eye)
            pair2_angle_batch_list.append(pair_angle)
            pair2_coord_batch_list.append(pair_coord)

            # hard for the 1st part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_group_batch(hard_batch_size, group_idx[1], inside_shuffle)
            pair1_eye_batch_list.append(pair_eye)
            pair1_angle_batch_list.append(pair_angle)
            pair1_coord_batch_list.append(pair_coord)
            # hard for the 2nd part
            pair_eye, pair_angle, pair_coord = self.gaze_data[index].next_group_batch(hard_batch_size, group_idx[2], inside_shuffle)
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

    def next_group_batch_pair_random(self, batch_size, inside_shuffle=True):
        pair1_eye_batch_list = []
        pair2_eye_batch_list = []
        pair1_angle_batch_list = []
        pair2_angle_batch_list = []
        pair1_coord_batch_list = []
        pair2_coord_batch_list = []
        sum_batch_size = 0

        # random select some persons
        num_select_person = len(self.gaze_data) // 3
        batch_size_one_person = batch_size // num_select_person + 1
        assert batch_size_one_person > 0

        idx_person = random.sample(range(len(self.gaze_data)), num_select_person)

        for index in idx_person:
            # calculate the batch size for each person
            if (sum_batch_size + batch_size_one_person) < batch_size:
                cur_batch_size = batch_size_one_person
            else:
                cur_batch_size = batch_size - sum_batch_size

            group_idx = random.sample(range(GazeData.num_groups), 2)
            assert cur_batch_size < 300 // GazeData.num_groups

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

