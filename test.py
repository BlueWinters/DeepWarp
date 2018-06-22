
import tensorflow as tf
import argparse
import numpy as np
import os
import cv2

from model import DeepGaze
import tools.helper as helper



def test(args):
    model_path = args.model_path
    num_frame = args.num_frame
    img_path = args.img_path
    tsf_type = args.tsf_type
    out_path = args.out_path

    if os.path.exists(out_path) == False:
        os.mkdir(out_path)

    # build graph
    deepgaze = DeepGaze(batch_size=1, name='deepwarp')
    deepgaze.build_graph()

    # restore all variables from model
    sess = tf.Session()
    saver = tf.train.Saver(deepgaze.variables)
    saver.restore(sess, '{}/model-99'.format(model_path))

    # load images
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # detect faces from another image
    el_img, el_fc, er_img, er_fc, os_l, os_r = helper.preprocess_image(rgb_img)
    # save sequence images
    images_list = []

    # get the angle as array
    # e_agl_array, e_re_agl_array = helper.get_eyes_horizontal_move_angle()
    e_agl_array, e_re_agl_array = helper.get_eyes_vertical_move_angle()

    # flip the right image to left
    er_img = er_img[:, :, ::-1, :]

    # actually, it generate the eyes sequences
    for n in range(num_frame):
        angle = e_agl_array[:,n,:]
        re_angle = e_re_agl_array[:,n,:]
        # output
        eye_left = sess.run(deepgaze.output,
                            feed_dict={deepgaze.input: el_img, deepgaze.is_train: False,
                                       deepgaze.angle: angle, deepgaze.re_angle: re_angle})

        eye_right = sess.run(deepgaze.output,
                            feed_dict={deepgaze.input: er_img, deepgaze.is_train: False,
                                       deepgaze.angle: angle, deepgaze.re_angle: re_angle})
        eye_right = eye_right[:, :, ::-1, :]
        # replace with our generated eyes
        new_face = np.copy(rgb_img)
        new_face = helper.replace(new_face, eye_left, os_l)
        new_face = helper.replace(new_face, eye_right, os_r)
        images_list.append(new_face)

    # save all the images
    helper.save_as_gif(images_list, out_path, 'h_0_0_all')





if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--is_left', type=bool, default=True)
    parse.add_argument('--model_path', type=str, default='save/new10')
    parse.add_argument('--out_path', type=str, default='save/new10/gif_v')
    parse.add_argument('--num_frame', type=int, default=30)
    parse.add_argument('--img_path', type=str, default='D:\deepwarp\deep-gaze-warp/backup/sequence\keep_left/1/14_5.png')
    parse.add_argument('--tsf_type', choices=[0,1,2], default=0, help='')

    # D:\deepwarp\deep-gaze-warp\sequence\keep_left\0\0_9.png
    # D:\deepwarp\deep-gaze-warp\sequence\keep_left\1\14_5.png
    # D:\deepwarp\deep-gaze-warp/backup/flow/1.jpg
    # D:\deepwarp\dataset\ColumbiaGaze/0029/0029_2m_0P_0V_0H.jpg
    # D:\deepwarp\dataset\ColumbiaGaze/0029/0029_2m_15P_-10V_5H.jpg

    args = parse.parse_args()
    test(args)