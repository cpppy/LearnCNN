import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
import printRes

digits = load_digits()
x_data = digits.data.astype(np.float32)
y_data = digits.target.astype(np.float32).reshape(-1, 1)

print(x_data.shape)
print(y_data.shape)

#printRes.printArray2Mat(x_data[0])
# print(y_data[0])

#----------------------------

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)
# print(x_data[0])


from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder().fit_transform(y_data).todense()
#print(y[0])

# ********************now, we get: x_data & y ********************


#transfer x_data to image_form
x = x_data.reshape(-1, 8, 8, 1)
#print(x[0])

batch_size = 8


def generateBatch(x, y, n_examples, batch_size):
    for i in range(n_examples // batch_size):
        start = i*batch_size
        end = start + batch_size
        batch_xs = x[start: end]
        batch_ys = y[start: end]
        yield batch_xs, batch_ys     #generate every batch


#-------------------------------part 0
tf.reset_default_graph()
#input layer
tf_x = tf.placeholder(tf.float32, [None, 8, 8, 1])  #input number-image
tf_y = tf.placeholder(tf.float32, [None, 10])    # 0~9  10numbers, y is probablity of every number

#-------------------------------part 1
#conv_layer  & active_layer

#filter ( conv core)
conv_filter_w1 = tf.Variable(tf.random_normal([3, 3, 1, 10]))
# size(height & width):3*3    deep(in_channels):1   quantityOfFilters: 10

conv_filter_b1 = tf.Variable(tf.random_normal([10]))
# 10 filters output 10 num, its need 10 bias_value

relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(tf_x, conv_filter_w1, strides=[1,1,1,1], padding="SAME") + conv_filter_b1)

#pooling layer
max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
# ksize(the size of pooling-figure) : [1, height, width, 1]
# strides(step-distance of pooling-figure): [1, heigh_dis, width_dis,1]

print("max_pool1:\n", max_pool1)

#----------------------------------part 2
#conv_layer2
conv_filter_w2 = tf.Variable(tf.random_normal([3,3,10,5]))
# generator five 3*3*10 filter-cores
conv_filter_b2 = tf.Variable(tf.random_normal([5]))
# generator five bias_value for each filter-core

conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 2, 1], padding='SAME') + conv_filter_b2
print(conv_out2)

#----------------------------------
#batch Normalization layer & active layer
batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
shift = tf.Variable(tf.zeros([5]))
scale = tf.Variable(tf.ones([5]))
epsilon = 1e-3
BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)
print(BN_out)
relu_BN_maps2 = tf.nn.relu(BN_out)


#pooling layer2
max_pool2 = tf.nn.max_pool(relu_BN_maps2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

max_pool2_flat = tf.reshape(max_pool2, [-1, 2*2*5])

#fc layer
fc_w1 = tf.Variable(tf.random_normal([2*2*5,50]))
fc_b1 =  tf.Variable(tf.random_normal([50]))
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)


#output layer
out_w1 = tf.Variable(tf.random_normal([50,10]))
out_b1 = tf.Variable(tf.random_normal([10]))
pred = tf.nn.softmax(tf.matmul(fc_out1,out_w1)+out_b1)

loss = -tf.reduce_mean(tf_y*tf.log(tf.clip_by_value(pred,1e-11,1.0)))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_pred = tf.argmax(pred, 1)
bool_pred = tf.equal(tf.argmax(tf_y, 1),y_pred)

accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000): # 迭代1000个周期
        for batch_xs, batch_ys in generateBatch(x,y,y.shape[0],batch_size): # 每个周期进行MBGD算法
            sess.run(train_step,feed_dict={tf_x:batch_xs,tf_y:batch_ys})
        if(epoch%1==0):
            res = sess.run(accuracy,feed_dict={tf_x:x,tf_y:y})
            print (epoch,res)
    res_ypred = y_pred.eval(feed_dict={tf_x:x,tf_y:y}).flatten() # 只能预测一批样本，不能预测一个样本
    print(res_ypred)
