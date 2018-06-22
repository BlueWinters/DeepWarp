
import tkinter as tk
import cv2
import os
import tensorflow as tf
import numpy as np
import tools.helper as helper

from PIL import Image, ImageTk
from version.ver_4.model import DeepGaze

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_input_from_image():
    global el_img, er_img, os_l, os_r, angle
    angle[:, 0] = float(scale_lh.get()) / 90.
    angle[:, 1] = float(scale_lv.get()) / 90.
    angle[:, 2] = float(scale_rh.get()) / 90.
    angle[:, 3] = float(scale_rv.get()) / 90.
    re_angle = np.zeros_like(angle)
    return el_img, er_img, angle, re_angle, os_l, os_r

def get_output_from_sess(el_img, er_img, angle, re_angle):
    global deepgaze, sess
    eye_left = sess.run(deepgaze.output,
                        feed_dict={deepgaze.input: el_img, deepgaze.is_train: False,
                                   deepgaze.angle: angle[:,0:2], deepgaze.re_angle: re_angle[:,0:2]})

    eye_right = sess.run(deepgaze.output,
                         feed_dict={deepgaze.input: er_img, deepgaze.is_train: False,
                                    deepgaze.angle: angle[:,2:4], deepgaze.re_angle: re_angle[:,2:4]})
    eye_right = eye_right[:, :, ::-1, :]
    return eye_left, eye_right

def generate(path):
    # path = entry.get()
    global cur_rgb_image
    if cur_rgb_image is not None:
        print('process......')
        el_img, er_img, angle, re_angle, os_l, os_r = get_input_from_image()
        el, er = get_output_from_sess(el_img, er_img, angle, re_angle)

        new_image = np.copy(cur_rgb_image)
        new_image = helper.replace(new_image, el, os_l)
        rgb_new_image = helper.replace(new_image, er, os_r)
        bgr_new_image = cv2.cvtColor(rgb_new_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('deepwarp', bgr_new_image)

        global label_img
        img_wapper = ImageTk.PhotoImage(Image.fromarray(rgb_new_image))
        label_img.configure(image=img_wapper)
        label_img.image = img_wapper
    else:
        print('no image......')

def load(entry):
    path = entry.get()
    if os.path.exists(path) == False:
        return

    global cur_rgb_image, cur_bgr_image
    cur_bgr_image = cv2.imread(path)
    cur_rgb_image = cv2.cvtColor(cur_bgr_image, cv2.COLOR_BGR2RGB)

    # resize_image = cv2.resize(cur_rgb_image, (640, 480))
    img_wapper = ImageTk.PhotoImage(Image.fromarray(cur_rgb_image))
    label_img.configure(image=img_wapper)
    label_img.image = img_wapper

    global el_img, er_img, angle, re_angle, os_l, os_r
    el_img, _, er_img, _, os_l, os_r = helper.preprocess_image(cur_rgb_image)
    er_img = er_img[:, :, ::-1, :]

    # cv2.imshow('deepwarp', cur_bgr_image)



if __name__ == '__main__':
    # Model
    model_path = 'D:\deepwarp\deep-gaze-warp/version/ver_4'
    deepgaze = DeepGaze(batch_size=1, name='deepwarp')
    deepgaze.build_graph()

    sess = tf.Session()
    saver = tf.train.Saver(deepgaze.variables)
    saver.restore(sess, '{}/model-99'.format(model_path))

    default = 'D:\deepwarp\deep-gaze-warp/backup/sequence\keep_left/1/14_5.png'
    cur_rgb_image, cur_bgr_image = None, None
    el_img, er_img, os_l, os_r = (None, ) * 4
    angle = np.zeros(shape=(1, 4), dtype=np.float32)
    scale_h, scale_v = None, None

    # GUI
    root = tk.Tk()
    root.title('deepwarp')
    # root.resizable(width=False, height=False)

    width, height = 1280, 720
    frm_img = tk.Frame(width=1280, height=720)
    frm_img.grid(row=0, column=0, padx=1, pady=1)
    frm_img.grid_propagate(0)
    frm_ctl = tk.Frame(width=640, height=250)
    frm_ctl.grid(row=1, column=0, padx=1, pady=1)

    label_img = tk.Label(frm_img)
    label_img.grid(row=0, column=0, padx=1, pady=1)

    label_path = tk.Label(frm_ctl, text='path')
    label_path.grid(row=0, column=0, padx=5, pady=5)

    default_var = tk.StringVar()
    default_var.set(default)
    entry_path = tk.Entry(frm_ctl, width=60, text=default_var, state='normal')
    entry_path.grid(row=0, column=1, padx=5, pady=5)

    btn_load = tk.Button(frm_ctl, text='load', command=lambda: load(entry_path))
    btn_load.grid(row=0, column=2)
    btn_gen = tk.Button(frm_ctl, text='generate', command=lambda: generate(entry_path))
    btn_gen.grid(row=0, column=3)

    scale_lv = tk.Scale(frm_ctl, from_=-30., to=30., resolution=0.5, length=200, orient=tk.HORIZONTAL, command=generate)
    scale_lv.grid(row=0, column=4)
    scale_lh = tk.Scale(frm_ctl, from_=-60., to=60., resolution=0.5, length=200, orient=tk.HORIZONTAL, command=generate)
    scale_lh.grid(row=0, column=5)

    scale_rv = tk.Scale(frm_ctl, from_=-30., to=30., resolution=0.5, length=200, orient=tk.HORIZONTAL, command=generate)
    scale_rv.grid(row=0, column=6)
    scale_rh = tk.Scale(frm_ctl, from_=-60., to=60., resolution=0.5, length=200, orient=tk.HORIZONTAL, command=generate)
    scale_rh.grid(row=0, column=7)



    root.mainloop()


