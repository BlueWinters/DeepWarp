
import scipy
import cv2
import numpy as np
import json
import scipy.io as sio
import os

from glob import glob


height, width = 41*2, 51*2
num_feat_pts = 7
angle_dim = 2




def expansion_to_layer(feat_pts):
    feat_pts_layers = np.zeros(shape=(height, width, num_feat_pts, 2), dtype=np.int)
    for n in range(num_feat_pts):
        for w in range(width):
            feat_pts_layers[:, w, n, 0] = w - feat_pts[n, 0]
        for h in range(height):
            feat_pts_layers[h, :, n, 1] = h - feat_pts[n, 1]
    feat_points_layers = np.reshape(feat_pts_layers, (height, width, num_feat_pts * 2))
    return feat_points_layers

def get_eye_local_coordinate(ldmks_of_all, image):
    eye_coord_global = np.zeros(shape=(num_feat_pts, 2), dtype=np.int)
    eye_coord_local = np.zeros(shape=(num_feat_pts, 2), dtype=np.int)
    x_max, x_min, y_max, y_min = (0, 0, 0, 0) # index
    ldmks_idx = [22, 2, 5, 8, 11, 14]
    ratio = 2.0 #

    for n, idx in enumerate(ldmks_idx):
        eye_coord_global[n, :] = ldmks_of_all[idx, :-1]
        x_max = n if eye_coord_global[n, 0] > eye_coord_global[x_max, 0] else x_max
        x_min = n if eye_coord_global[n, 0] < eye_coord_global[x_min, 0] else x_min
        y_max = n if eye_coord_global[n, 1] > eye_coord_global[y_max, 1] else y_max
        y_min = n if eye_coord_global[n, 1] < eye_coord_global[y_min, 1] else y_min

    # index --> coordinate
    x_max = eye_coord_global[x_max, 0]
    x_min = eye_coord_global[x_min, 0]
    y_max = eye_coord_global[y_max, 1]
    y_min = eye_coord_global[y_min, 1]

    # exclude the center point, because it hadn't be assign value
    x_cen = int(np.mean(eye_coord_global[:-1, 0], dtype=np.int))
    y_cen = int(np.mean(eye_coord_global[:-1, 1], dtype=np.int))
    half_w = int((x_max - x_min) / 2 * ratio + 0.5)
    half_h = int(half_w * 0.8 / 2 * ratio + 0.5)

    new_x_org = x_cen - half_w
    new_y_org = y_cen - half_h
    warp_part = image[(y_cen - half_h):(y_cen + half_h), (x_cen - half_w):(x_cen + half_w)]
    resize_warp = cv2.resize(warp_part, (width, height)) # cv2?

    def draw_landmark(image, point, radius, flag=True):
        cv2.circle(image, point, radius, (255, 255, 255), -1)

    for n in range(0, num_feat_pts-1):
        local_x, local_y = eye_coord_global[n, 0] - new_x_org, eye_coord_global[n, 1] - new_y_org
        eye_coord_local[n, 0] = int(local_x * width / (2 * half_w))  # 51?
        eye_coord_local[n, 1] = int(local_y * height / (2 * half_h))  # 41?
        # draw_landmark(resize_warp, (eye_coord_local[n, 0], eye_coord_local[n, 1]), 2)
        # draw_landmark(warp_part, (local_x, local_y), 5,)

    # assign the value to the center point
    local_x, local_y = x_cen - new_x_org, y_cen - new_y_org
    eye_coord_local[-1, 0] = int(local_x * width / (2 * half_w))  # 51?
    eye_coord_local[-1, 1] = int(local_y * height / (2 * half_h))  # 41?
    # draw_landmark(resize_warp, (eye_coord_local[-1, 0], eye_coord_local[-1, 1]), 2)
    # draw_landmark(warp_part, (local_x, local_y), 5)

    return eye_coord_local, resize_warp


def process_one_folder_to_mat(in_path, out_path):
    json_fns = glob('{}/*.json'.format(in_path))
    N = len(json_fns)

    eye_images = np.zeros((N, height, width, 3), np.uint8)
    angle_mat = np.zeros((N, angle_dim), np.float32)
    feat_points = np.zeros((N, height, width, num_feat_pts * 2), np.int)

    for i, json_fn in enumerate(json_fns):
        file_name = '{}.jpg'.format(json_fn[:-5])
        print('process image: {}'.format(file_name))
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = json.load(open(json_fn, 'r'))

        # internal function
        def process_json_list(json_list):
            ldmks = [eval(s) for s in json_list]
            return np.array([(x, image.shape[0] - y, z) for (x, y, z) in ldmks])

        # read json data
        ldmks_interior_margin = process_json_list(data['interior_margin_2d'])
        ldmks_caruncle = process_json_list(data['caruncle_2d'])
        ldmks_iris = process_json_list(data['iris_2d'])
        ldmks_of_all = np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]])

        # process coordinates of landmarks and images
        eye_coord_local, resize_warp = get_eye_local_coordinate(ldmks_of_all, image)
        eye_coord_layer = expansion_to_layer(eye_coord_local)
        # process angles
        angle = eval(data['eye_details']['look_vec'])[:2]

        # store
        eye_images[i,:,:,:] = resize_warp
        angle_mat[i,:] = angle
        feat_points[i,:,:,:] = eye_coord_layer

    # save to mat
    print('process sum images {}'.format(N))
    data = {'eye_images': eye_images[:N, ...].astype(np.uint8),
            'feat_points': feat_points[:N, ...].astype(np.int8),
            'angle_mat': angle_mat[:N, ...].astype(np.float32)}
    sio.savemat(out_full_path, data)



if __name__ == '__main__':
    root_path = 'D:/UnityEyes_Windows/all'
    out_path = 'D:/UnityEyes_Windows/all/mat3'
    path_range = list(range(1, 31))
    for path in path_range:
        in_full_path = '{}/{:04d}'.format(root_path, path)
        out_full_path = '{}/{:04d}.mat'.format(out_path, path)
        process_one_folder_to_mat(in_full_path, out_full_path)