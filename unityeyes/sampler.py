
import autopy
import math
import time
import random
import os
import shutil
import time
import win32gui
import win32con
import win32api


KEY_SAVE = 83
KEY_RANDOM = 82
KEY_LIGHT = 76


def sample_one_person(n, num_x=5, num_y=5):
    save_path = 'D:/UnityEyes_Windows/imgs'
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    # reset
    win32gui.SendMessage(handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
    center_x = (clt_left + clt_right) // 2
    center_y = (clt_top + clt_bottom) // 2
    win32api.SetCursorPos([center_x, center_y])

    # press 'L'
    win32api.keybd_event(KEY_LIGHT, 0, 0, 0)  # key down
    time.sleep(1)
    win32api.keybd_event(KEY_LIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)  # key up
    # press 'R'
    win32api.keybd_event(KEY_RANDOM, 0, 0, 0)  # key down
    time.sleep(1)
    win32api.keybd_event(KEY_RANDOM, 0, win32con.KEYEVENTF_KEYUP, 0)  # key up

    # number of points for vertical and horizontal
    # num_x, num_y = 5, 5

    step_x, step_y = width // (num_x + 1), height // (num_y + 1)
    for i in range(1, num_y+1):
        for j in range(1, num_x+1):
            x = clt_left + j * step_x
            y = clt_top + i * step_y
            print('{},{}'.format(x, y))
            win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
            win32api.SetCursorPos([x, y])
            win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
            time.sleep(0.5)
            win32api.keybd_event(KEY_SAVE, 0, 0, 0) # key down
            win32api.keybd_event(KEY_SAVE, 0, win32con.KEYEVENTF_KEYUP, 0)  # key up


if __name__ == '__main__':
    width, height = 640, 480
    handle = win32gui.FindWindow('UnityWndClass', None)
    win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(handle)
    win_left += 2
    title_height = win_bottom - win_top - height
    clt_left, clt_top, clt_right, clt_bottom = win_left, win_top + title_height, win_right, win_bottom

    window_text = win32gui.GetWindowText(handle)
    class_name = win32gui.GetClassName(handle)
    print("handle: {:x}, {:d}".format(handle, handle))
    print('position: {},{},{},{}'.format(clt_left, clt_top, clt_right, clt_bottom))
    print('width:{}, height:{}'.format(clt_right - clt_left, clt_bottom - clt_top))
    print('class name: {}'.format(win32gui.GetClassName(handle)))

    win32gui.SendMessage(handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)

    for n in range(21, 31):
        print('person id: {}'.format(n))
        sample_one_person(n, 20, 15)
        time.sleep(1)
        os.rename('D:/UnityEyes_Windows/imgs', 'D:/UnityEyes_Windows/{:04d}'.format(n))
        shutil.move('D:/UnityEyes_Windows/{:04d}'.format(n), 'D:/UnityEyes_Windows/all')
        time.sleep(1)