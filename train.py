
import tensorflow as tf
import argparse
import time

from model import DeepGaze
from tools import gaze_synthesis
import tools.helper as helper



def train(args):
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    save_path = args.save_path
    scheduler = args.scheduler
    data_path = args.data_path
    summary_step = args.summary_step
    is_resume = args.is_resume

    # build graph
    deepgaze = DeepGaze(batch_size=batch_size, name='deepwarp')
    deepgaze.build_graph()

    learn_rate = tf.placeholder(dtype=tf.float32, name='lr')

    net_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, deepgaze.name)
    net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, deepgaze.name)
    with tf.control_dependencies(net_update_ops):
        solver = tf.train.AdamOptimizer(learn_rate).minimize(deepgaze.loss, var_list=net_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    data = gaze_synthesis.SynthesisGaze()
    data.load_synthesis_gaze(data_path=data_path)

    if summary_step > 0:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(save_path, sess.graph)
    else:
        merged, writer = None, None

    file = open('{}/train.txt'.format(save_path), 'w')
    scheduler = helper.learn_rate_scheduler(scheduler)
    weights = helper.learn_rate_scheduler('tools/weights')
    saver = tf.train.Saver(tf.global_variables())
    lr, wc = 0., 0.

    if is_resume is not None:
        saver.restore(sess, is_resume)

    # training
    try:
        for epochs in range(num_epochs):
            num_iter = 100
            average_loss, start = 0, time.time()
            average_c_loss, average_f_loss = 0, 0
            lr = scheduler[epochs] if epochs in scheduler else lr
            wc = weights[epochs] if epochs in weights else wc
            batch_x, batch_re_x, batch_angle, batch_re_angle, batch_feat_coord, batch_re_feat_coord = (None,)*6

            # training
            for iter in range(num_iter):
                batch_x, batch_re_x, batch_angle, batch_re_angle, batch_feat_coord, batch_re_feat_coord \
                    = data.next_group_batch_pair_random(batch_size)

                _, m_loss, c_loss, f_loss = sess.run([
                    solver, deepgaze.loss, deepgaze.coarse_loss, deepgaze.fine_loss],
                    feed_dict={deepgaze.input: batch_x, deepgaze.re_input: batch_re_x,
                               deepgaze.angle: batch_angle, deepgaze.re_angle: batch_re_angle,
                               deepgaze.is_train: True, learn_rate: lr, deepgaze.coarse_coef: wc})
                average_loss += m_loss / num_iter
                average_c_loss += c_loss / num_iter
                average_f_loss += f_loss / num_iter

            # summary
            if summary_step > 0 and epochs % summary_step == 0:
                summary = sess.run(merged,
                                   feed_dict={deepgaze.input: batch_x, deepgaze.re_input: batch_re_x,
                                              deepgaze.angle: batch_angle, deepgaze.re_angle: batch_re_angle,
                                              deepgaze.is_train: False, deepgaze.coarse_coef: wc})
                writer.add_summary(summary, epochs)
                saver.save(sess, '{}/model'.format(save_path), epochs)

            elapsed = time.time() - start
            liner = 'epoch {:3d}/{:3d}, sum_loss {:.6f}, loss_c {:.6f}, loss_f {:.6f}, time {:.6f}'.format(
                epochs, num_epochs, average_loss, average_c_loss, average_f_loss, elapsed)
            print(liner), file.write(liner+'\n')
            file.flush()
    except KeyboardInterrupt:
        # capture the Ctrl+C
        print('interrupt by Ctrl+C, save the model...')
        saver.save(sess, '{}/model'.format(save_path))
        exit()

    # save model
    saver.save(sess, '{}/model'.format(save_path))
    # close all
    file.close()
    sess.close()





if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--is_resume', type=str, default=None)
    parse.add_argument('--is_left', type=bool, default=True)
    parse.add_argument('--data_path', type=str, default='D:/UnityEyes_Windows/all/mat')
    parse.add_argument('--save_path', type=str, default='save/new14')
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--num_epochs', type=int, default=100)
    parse.add_argument('--scheduler', type=str, default='tools/scheduler')
    parse.add_argument('--summary_step', type=int, default=1)

    args = parse.parse_args()
    train(args)
