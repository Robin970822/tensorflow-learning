#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:23:55 2018

@author: hanxy
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

def addConv2d(inputData,rows,cols,inChannel,outChannel,rowStep,colStep):
    W=tf.Variable(tf.truncated_normal([rows,cols,inChannel,outChannel],stddev=0.1))#标准差不能设置太大,否则e的次方太大=nan
    b=tf.Variable(tf.constant(0.1, shape=[outChannel]))
    W_conv1=tf.nn.conv2d(inputData, W, strides=[1, rowStep, colStep, 1], padding='SAME')
    return tf.nn.relu(W_conv1 + b)

def addPoll2x2(inputData,batch, rows , cols, channels, rowStep, colStep):
    # [batch, rows , cols, channels] and type tf.float32  [1,2,2,1]
    return tf.nn.max_pool(inputData, ksize=[batch, rows , cols, channels],strides=[1, rowStep, colStep, 1], padding='SAME')


def add_Layer(inputData,inputSize,outputSize,isSoftMax=False,activationFun=tf.nn.relu):
    W=tf.Variable(tf.truncated_normal([inputSize,outputSize],stddev=0.1))#每一列代表下一个节点对应的所有权重
    b=tf.Variable(tf.constant(0.1, shape=[outputSize]))
    WPlusInAddb=tf.matmul(inputData,W)+b
    if activationFun is None:
        outputData=WPlusInAddb;
    else:
        outputData=activationFun(WPlusInAddb);
    if isSoftMax is True:
        outputData=tf.nn.softmax(outputData);
    return outputData;

mnist = input_data.read_data_sets("MNIST/", one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(x, [-1,28,28,1])

conv1=addConv2d(x_image,5,5,1,32,1,1)
poll1=addPoll2x2(conv1,1,2,2,1,2,2)
conv2=addConv2d(poll1,5,5,32,64,1,1)
poll2=addPoll2x2(conv2,1,2,2,1,2,2)

poll2Data=tf.reshape(poll2,[-1,7*7*64])
fnn1=add_Layer(poll2Data,7*7*64,1024)
keep_prob = tf.placeholder(tf.float32)
drop1 = tf.nn.dropout(fnn1, keep_prob)

Ws=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))#每一列代表下一个节点对应的所有权重
bs=tf.Variable(tf.constant(0.1, shape=[10]))
WPlusInAddbs=tf.matmul(drop1,Ws)+bs;
softmaxOut=tf.nn.softmax(WPlusInAddbs);

#softmaxOut=add_Layer(drop1,1024,10,True,None)
cross_entropy = -tf.reduce_sum(y*tf.log(softmaxOut))
#cross_entropy=tf.square(softmaxOut-y);
#cross_entropy=tf.reduce_mean(tf.reduce_sum(cross_entropy,reduction_indices=[1]))
#tf.clip_by_value(softmaxOut,1e-10,1.0)
#cross_entropy = -tf.reduce_sum(y*tf.log(softmaxOut))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(softmaxOut,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session=tf.Session()
session.run(tf.global_variables_initializer())

for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
#      s=session.run(softmaxOut,feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0});
#      print len(batch[1]);
#      print(session.run(softmaxOut,feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0}))
#      print(session.run(WPlusInAddbs,feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0}))
#      print(session.run(softmaxOut,feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0}))
      train_accuracy = session.run(accuracy,feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
      print "Iteration %d, training accuracy %g"%(i, train_accuracy)
  session.run(train_step,feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
  
accuracy = session.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
print "test accuracy %g"%accuracy



#prediction=add_Layer(x,784,10,True)
##prediction=add_Layer(prediction1,200,10,True);
#loss = -tf.reduce_sum(y*tf.log(prediction))
##loss=tf.square(prediction-y);
##loss=tf.reduce_mean(tf.reduce_sum(loss,reduction_indices=[1]))
#train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#init=tf.global_variables_initializer()
#session=tf.Session()
#session.run(init)
#for i in range(10000):
#    x_data, y_data = mnist.train.next_batch(100)
#    session.run(train_step,feed_dict={x:x_data,y:y_data})
#    if i%50==0:
#        print(session.run(loss,feed_dict={x:x_data,y:y_data}))
#correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#W=tf.Variable(tf.random_normal([784,10]))#每一列代表下一个节点对应的所有权重
#b=tf.Variable(tf.zeros([10])+0.1)

