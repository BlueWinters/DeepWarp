
import numpy as np
import dlib
import cv2
import os
import scipy.misc as misc
import imageio



width = 51*1
height = 41*1
angle_dim = 2

index_left_eye = [42, 43, 44, 45, 46, 47]
index_right_eye = [36, 37, 38, 39, 40, 41]
num_feature_point = 7
detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'D:\deepwarp\deep-gaze-warp\data\model\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)

def learn_rate_scheduler(file_name):
    scheduler = {}
    file = open(file_name, 'r')
    for liner in file:
        info = liner.split(',')
        epoch = int(info[0])
        lr = float(info[1])
        scheduler[epoch] = lr
    return scheduler

def expansion_to_layer(feat_points):
    feat_points_layers = np.zeros(shape=(height, width, num_feature_point, 2), dtype=np.int)
    for n in range(num_feature_point):
        for w in range(width):
            feat_points_layers[:, w, n, 0] = w - feat_points[n, 0]
        for h in range(height):
            feat_points_layers[h, :, n, 1] = h - feat_points[n, 1]
    feat_points_layers = np.reshape(feat_points_layers, (height, width, num_feature_point*2))
    return feat_points_layers

def process_one_eye(shape, eye_index, image):
    eye_coord_global = np.zeros(shape=(num_feature_point, 2), dtype=np.int)
    eye_coord_local = np.zeros(shape=(num_feature_point, 2), dtype=np.int)
    x_max, x_min, y_max, y_min = (0, 0, 0, 0) # index
    ratio = 2 #

    for n in range(len(eye_index)):
        eye_coord_global[n, 0] = shape.part(eye_index[n]).x
        eye_coord_global[n, 1] = shape.part(eye_index[n]).y
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

    for n in range(0, num_feature_point-1):
        local_x, local_y = eye_coord_global[n, 0] - new_x_org, eye_coord_global[n, 1] - new_y_org
        eye_coord_local[n, 0] = int(local_x * width / (2 * half_w))  # 51?
        eye_coord_local[n, 1] = int(local_y * height / (2 * half_h))  # 41?

    # assign the value to the center point
    local_x, local_y = x_cen - new_x_org, y_cen - new_y_org
    eye_coord_local[-1, 0] = int(local_x * width / (2 * half_w))  # 51?
    eye_coord_local[-1, 1] = int(local_y * height / (2 * half_h))  # 41?

    return eye_coord_local, resize_warp, (x_cen, y_cen, half_w, half_h)

def preprocess_image(rgb_img):
    global detector
    global predictor

    detection = detector(rgb_img, 1)
    for det in detection:
        shape = predictor(rgb_img, det)
        # get relative information
        left_eye_coord_local, resize_left_img, out_shape_left = process_one_eye(shape, index_left_eye, rgb_img)
        right_eye_coord_local, resize_right_img, out_shape_right = process_one_eye(shape, index_right_eye, rgb_img)
        # expand local coordinate
        left_eye_coord_local = expansion_to_layer(left_eye_coord_local)
        right_eye_coord_local = expansion_to_layer(right_eye_coord_local)
        # expand dimension
        left_eye_coord_local = np.expand_dims(left_eye_coord_local, axis=0)
        resize_left_img = np.expand_dims(resize_left_img, axis=0)
        right_eye_coord_local = np.expand_dims(right_eye_coord_local, axis=0)
        resize_right_img = np.expand_dims(resize_right_img, axis=0)

        # normalize the value
        resize_left_img = resize_left_img.astype(np.float) / 255.
        resize_right_img = resize_right_img.astype(np.float) / 255.
        for n in range(num_feature_point):
            left_eye_coord_local[:, :, :, 2*n+0] = left_eye_coord_local[:, :, :, 2*n+0] / width
            left_eye_coord_local[:, :, :, 2*n+1] = left_eye_coord_local[:, :, :, 2*n+1] / height
            right_eye_coord_local[:, :, :, 2*n+1] = right_eye_coord_local[:, :, :, 2*n+1] / width
            right_eye_coord_local[:, :, :, 2*n+1] = right_eye_coord_local[:, :, :, 2*n+1] / height
        # for only one face
        return resize_left_img, left_eye_coord_local, \
               resize_right_img, right_eye_coord_local, \
               out_shape_left, out_shape_right

def get_eyes_angle_list(num_horizontal=30, num_vertical=10):
    h_degree = np.array(list(range(-30, 31, 1)), np.float)
    v_degree = np.array(list(range(-10, 11, 1)), np.float)

    x = np.linspace(-30, 30, num_horizontal, dtype=np.float)
    y = np.linspace(-10, 10, num_vertical, dtype=np.float)
    h_code, v_code = np.meshgrid(x, y)
    h_value, v_value = h_code / 90., v_code / 90.

def get_eyes_vertical_move_angle(num_frame=30):
    v_code = np.linspace(-30., 30., num_frame, dtype=np.float)
    h_code = np.linspace(-0., 0., num_frame, dtype=np.float)
    h_value, v_value = h_code / 90., v_code / 90.
    angle = np.stack([h_value, v_value], axis=1)
    angle = np.expand_dims(angle, axis=0)
    return angle, np.zeros_like(angle)

def get_eyes_horizontal_move_angle(num_frame=30):
    h_code = np.linspace(-30., 30., num_frame, dtype=np.float)
    v_code = np.linspace(-0., 0., num_frame, dtype=np.float)
    h_value, v_value = h_code / 90., v_code / 90.
    angle = np.stack([h_value, v_value], axis=1)
    angle = np.expand_dims(angle, axis=0)
    return angle, np.zeros_like(angle)

def replace_eyes(image, out_eyes, out_shape, out_path, n):
    x_cen, y_cen, half_w, half_h = out_shape
    copy_image = np.copy(image)
    print(image.shape)
    print(out_eyes.shape) # 41,51
    out_eyes = np.squeeze(out_eyes, axis=0)

    # resize and save as eyes only
    # replace = cv2.resize(out_eyes, (51,41)) #(400, 250)
    save_path_and_name = os.path.join(out_path, '{}.jpg'.format(n))
    misc.imsave(save_path_and_name, out_eyes)

    resize_replace = cv2.resize(out_eyes, (2*half_w, 2*half_h)) * 255 # resize to original
    # resize_replace = np.transpose(resize_replace, axes=(1, 0, 2))
    copy_image[(y_cen - half_h):(y_cen + half_h), (x_cen - half_w):(x_cen + half_w), :] = resize_replace.astype(np.uint8)
    image_save_path_and_name = os.path.join(out_path, 'face_{}.jpg'.format(n))
    # print(image_save_path_and_name)
    misc.imsave(image_save_path_and_name, copy_image)
    return None

def save_as_gif(images_list, out_path, gif_file_name='all', save_image=False):
    if os.path.exists(out_path) == False:
        os.mkdir(out_path)

    # save as .png
    if save_image == True:
        for n in range(len(images_list)):
            file_name = '{}.png'.format(n)
            save_path_and_name = os.path.join(out_path, file_name)
            misc.imsave(save_path_and_name, images_list[n])
    # save as .gif
    out_path_and_name = os.path.join(out_path, '{}.gif'.format(gif_file_name))
    imageio.mimsave(out_path_and_name, images_list, 'GIF', duration=0.1)

def replace(image, out_eyes, out_shape):
    x_cen, y_cen, half_w, half_h = out_shape
    print('{},{},{},{}'.format(y_cen - half_h, y_cen + half_h, x_cen - half_w, x_cen + half_w))
    out_eyes = np.squeeze(out_eyes, axis=0)
    resize_replace = cv2.resize(out_eyes, (2 * half_w, 2 * half_h)) * 255  # resize to original
    image[(y_cen - half_h):(y_cen + half_h), (x_cen - half_w):(x_cen + half_w), :] = resize_replace.astype(np.uint8)
    return image