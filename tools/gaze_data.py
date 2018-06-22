
import numpy as np
import os
import random
import scipy.io as sio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class GazeData:
    # static member of the class
    num_groups = 2

    def __init__(self):
        self.eye_images = None
        self.feat_coord = None
        self.angles = None
        self.num_examples = 0
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def load_synthesis_gaze(self, data_path=None):
        assert self.eye_images == None
        print('extract gaze data: [left eye] from {}'.format(data_path))
        data = sio.loadmat(data_path)
        eye_images = data['eye_images'].astype(np.float32)
        angle = data['angle_mat'].astype(np.float32)
        feat_coord = data['feat_points'].astype(np.float32)

        # each pixel is limited between [0,255]
        eye_images = eye_images / 255.
        # landmark point coordinate(local, H:[-41,41],W:[51,51])
        num_features = int(np.shape(feat_coord)[-1] / 2)
        for n in range(num_features):
            feat_coord[:, :, :, 2 * n + 0] = feat_coord[:, :, :, 2 * n + 0] / 51.
            feat_coord[:, :, :, 2 * n + 1] = feat_coord[:, :, :, 2 * n + 1] / 41.

        # keep the matrix
        # print('number of data: eye images {:d}, angle encoded {:d}, feature coordinate {:d}'
        #       .format(np.shape(eye_images)[0], np.shape(angle)[0], np.shape(feat_coord)[0]))
        self.eye_images = eye_images
        self.angles = angle
        self.feat_coord = feat_coord
        self.num_examples = np.shape(self.eye_images)[0]

    def load_columbia_gaze(self, is_left=True, data_path=None):
        assert self.eye_images == None
        if is_left == True:
            print('extract gaze data: [left eye] from {}'.format(data_path))
            data = sio.loadmat(os.path.join(data_path))
            eye_images = data['left_eye_images'].astype(np.float32)
            angle = data['angle_mat'].astype(np.float32)
            feat_coord = data['left_feat_points'].astype(np.float32)
        else:
            print('extract gaze data: [right eye] from {}'.format(data_path))
            data = sio.loadmat(os.path.join(data_path))
            eye_images = data['right_eye_images'].astype(np.float32)
            angle = data['angle_mat'].astype(np.float32)
            feat_coord = data['right_feat_points'].astype(np.float32)

        if is_left == False:
            # flip the right image to left
            eye_images = eye_images[:, :, ::-1, :]
            # the horizontal angle
            angle[:, 1] = 0. - angle[:, 1]
            # the feature coordinate
            feat_coord = feat_coord[:, :, ::-1, :]

        # each pixel is limited between [0,255]
        eye_images = eye_images / 255.
        # vertical(0°, ±10°) and horizontal(0°, ±5°, ±10°, ±15°)
        angle = angle / 90.
        # landmark point coordinate(local, H:[-41,41],W:[51,51])
        # details can reference to: preprocess.py --> expansion_to_layer function
        num_features = int(np.shape(feat_coord)[-1] / 2)
        for n in range(num_features):
            feat_coord[:, :, :, 2*n+0] = feat_coord[:, :, :, 2*n+0] / 51.
            feat_coord[:, :, :, 2*n+1] = feat_coord[:, :, :, 2*n+1] / 41.

        # keep the matrix
        # print('number of data: eye images {:d}, angle encoded {:d}, feature coordinate {:d}'
        #       .format(np.shape(eye_images)[0], np.shape(angle)[0], np.shape(feat_coord)[0]))
        self.eye_images = eye_images
        self.angles = angle
        self.feat_coord = feat_coord
        self.num_examples = np.shape(self.eye_images)[0]

    def load_self_gaze(self, is_left=True, data_path=None):
        assert self.eye_images == None
        if is_left == True:
            print('extract gaze data: [left eye] from {}'.format(data_path))
            data = sio.loadmat(os.path.join(data_path))
            eye_images = data['left_eye_images'].astype(np.float32)
            angle = data['angle_mat'].astype(np.float32)
            feat_coord = data['left_feat_points'].astype(np.float32)
        else:
            print('extract gaze data: [right eye] from {}'.format(data_path))
            data = sio.loadmat(os.path.join(data_path))
            eye_images = data['right_eye_images'].astype(np.float32)
            angle = data['angle_mat'].astype(np.float32)
            feat_coord = data['right_feat_points'].astype(np.float32)

        # flip left and right
        if is_left == False:
            eye_images = eye_images[:,:,::-1,:]
            angle[:,1] = 9 - angle[:,1]
            feat_coord = feat_coord[:,:,::-1,:]

        # each pixel is limited between [0,255]
        eye_images = eye_images / 255.
        # vertical(0:9) and horizontal(0:9)
        angle = angle / 9. - 0.5
        # landmark point coordinate(local, H:[-41,41],W:[51,51])
        # details can reference to: preprocess.py --> expansion_to_layer function
        num_features = int(np.shape(feat_coord)[-1] / 2)
        for n in range(num_features):
            feat_coord[:, :, :, 2 * n + 0] = feat_coord[:, :, :, 2 * n + 0] / 51.
            feat_coord[:, :, :, 2 * n + 1] = feat_coord[:, :, :, 2 * n + 1] / 41.

        # keep the matrix
        # print('number of data: eye images {:d}, angle encoded {:d}, feature coordinate {:d}'
        #       .format(np.shape(eye_images)[0], np.shape(angle)[0], np.shape(feat_coord)[0]))
        self.eye_images = eye_images
        self.angles = angle
        self.feat_coord = feat_coord
        self.num_examples = np.shape(self.eye_images)[0]

    def get_batch_by_index(self, batch_size, idx_group, shuffle=True):
        ave_group_count = self.num_examples // GazeData.num_groups
        beg = idx_group*ave_group_count
        end = (idx_group+1)*ave_group_count if idx_group < GazeData.num_groups-1 else self.num_examples
        perm = np.array(random.sample(range(beg,end), batch_size), dtype=np.int)
        if shuffle == True:
            np.random.shuffle(perm)
        eyes = self.eye_images[perm]
        angles = self.angles[perm]
        feat_coord = self.feat_coord[perm]
        return eyes, angles, feat_coord

    def next_group_batch(self, batch_size, idx_group, shuffle=True):
        assert batch_size < self.num_examples // GazeData.num_groups
        assert idx_group < GazeData.num_groups # begin from 0
        pair_eye, pair_angle, pair_coord = self.get_batch_by_index(batch_size, idx_group, shuffle)
        return pair_eye, pair_angle, pair_coord

    def next_batch(self, batch_size, shuffle=True):
        start = self.index_in_epoch

        # Shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.num_examples)
            np.random.shuffle(perm0)
            self.eye_images = self.eye_images[perm0]
            self.angles = self.angles[perm0]
            self.feat_coord = self.feat_coord[perm0]

        # Go to the next epoch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            eye_images_rest_part = self.eye_images[start:self.num_examples]
            angles_rest_part = self.angles[start:self.num_examples]
            feat_coord_rest_part = self.feat_coord[start:self.num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.num_examples)
                np.random.shuffle(perm)
                self.eye_images = self.eye_images[perm]
                self.angles = self.angles[perm]
                self.feat_coord = self.feat_coord[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            eye_images_new_part = self.eye_images[start:end]
            angles_new_part = self.angles[start:end]
            feat_coord_new_part = self.feat_coord[start:end]
            return np.concatenate((eye_images_rest_part, eye_images_new_part), axis=0), \
                   np.concatenate((angles_rest_part, angles_new_part), axis=0), \
                   np.concatenate((feat_coord_rest_part, feat_coord_new_part), axis=0)
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.eye_images[start:end], self.angles[start:end], self.feat_coord[start:end]

    def is_iter_over(self, batch_size):
        return (self.index_in_epoch + batch_size) > self.num_examples



