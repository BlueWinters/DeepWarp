
import matplotlib.pyplot as plt
import numpy as np



def split_train_txt(full_file_path):
    loss, coarse, fine = [], [], []
    file = open(full_file_path, 'r')
    for liner in file:
        if 'epoch' in liner:
            all = liner.split(',')
            loss.append(float(all[1].split(' ')[-1]))
            coarse.append(float(all[2].split(' ')[-1]))
            fine.append(float(all[3].split(' ')[-1]))
    file.close()

    np_loss = np.array(loss, np.float32)
    np_coarse = np.array(coarse, np.float32)
    np_fine = np.array(fine, np.float32)
    return np_loss, np_coarse, np_fine

def plot_vggnet_performance(number, name):
    ver1 = split_train_txt('D:\deepwarp\DeepWarp\save/ver_1/train.txt')
    ver2 = split_train_txt('D:\deepwarp\DeepWarp\save/ver_2/train.txt')

    x = np.array(range(400))
    plt.plot(x, ver1[number][:400])
    plt.plot(x, ver2[number][:400])
    plt.title(name)
    plt.legend(('ver1', 'ver2'), loc='upper right')
    plt.show()


if __name__ == '__main__':
    plot_vggnet_performance(2, 'accuracy')