from __future__ import division

import tensorflow as tf
import numpy as np
import time
import cifar10
import cifar10_input



def variable_with_weight_loss(shape, stddev, wl):
    '''
    使用tf.truncated_normal截断的正态分布来初始化权重,这里给weight加一个L2的loss.
    我们使用wl控制L2 loss的大小,使用tf.nn.l2_loss计算weight的L2 loss.
    再使用tf.multiply让L2 loss乘wl,得到最后的weight loss,最后将weight loss添加到一个collection.便于后期优化
    :param shape:
    :param stddev:
    :param wl:
    :return:
    '''
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
    '''
    使用tf.nn.sparse_softmax_cross_entropy_with_logits将softmax和cross_entropy_loss计算合在一起
    并计算cross_entropy的均值添加到losses集合.以便于后面输出所有losses
    :param logits:
    :param labels:
    :return:
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train():
    max_steps = 30000
    batch_size = 128
    data_dir = './cifar-10-binary/cifar-10-batches-bin'

    '''
    使用cifar10_input类中的disorted_inputs函数产生训练数据,产生的数据是已经封装好的Tensor,每次会产生batch_size个
    这个函数已经对图片数据做了增强操作(随机水平翻转/剪切/随机对比度等)
    同时因为对图像处理需要耗费大量计算资源,该函数使用了16个独立的线程来加速任务,
    函数内部会产生线程池,使用会通过TensorFlow queue进行调度
    '''
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)




    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

    # 第一层 卷积-->池化-->lrn
    # 不带L2正则项(wl=0)的64个5x5x3的滤波器,
    # 使用lrn是从局部多个卷积核的响应中挑选比较大的反馈变得相对最大,并抑制其他反馈小的,增加模型泛化能力
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
    kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))  # bias直接初始化为0
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 第二层  卷积-->lrn-->池化
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
    kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层 使用全连接层 reshape后获取长度并创建FC1层的权重(带L2正则化)
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

    # 第四层 FC2层  节点数减半  依旧带L2正则
    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

    # 最后一层 这层的weight设为正态分布标准差设为上一个FC层的节点数的倒数
    # 这里我们不计算softmax,把softmax放到后面计算
    weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
    bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
    logits = tf.add(tf.matmul(local4, weight5), bias5)

    # 损失函数为两个带L2正则的FC层和最后的转换层
    # 优化器依旧是AdamOptimizer,学习率是1e-3
    losses = loss(logits, label_holder)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(losses)

    # in_top_k函数求出输出结果中top k的准确率,这里选择输出top1
    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

    # 创建默认session,初始化变量
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 启动图片增强线程队列
    tf.train.start_queue_runners()

    # 训练
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, losses], feed_dict={image_holder: image_batch,
                                                                label_holder: label_batch})
        duration = time.time() - start_time

        if step % 10 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

    # 评估模型的准确率,测试集一共有10000个样本
    # 我们先计算大概要多少个batch能测试完所有样本
    num_examples = 10000
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size  # 除去不够一个batch的
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                      label_holder: label_batch})
        true_count += np.sum(predictions)  # 利用top_k_op计算输出结果
        step += 1

    precision = true_count / total_sample_count
    print('precision @ 1=%.3f' % precision)  # 这里如果输出为0.00 是因为整数/整数  记得要导入float除法


if __name__ == '__main__':

    train()