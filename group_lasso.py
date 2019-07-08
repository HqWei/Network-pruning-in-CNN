import os
import numpy as np
import tensorflow as tf

'''
    Filter_lasso function
    Input:The weight of conv layer
    Output: loss+group_loss
'''
def weight_loss(loss1,conv11_filter,conv12_filter,conv13_filter,conv21_filter,conv22_filter,conv23_filter):
    # def filter_lasso(weight_pow):
    #     lasso = weight_pow.sum(dim=3).sum(dim=1).sum(dim=1).pow(1/2.).sum()
    #     return lasso
    # def channel_lasso(weight_pow):
    #     lasso = weight_pow.sum(dim=2).sum(dim=0).sum(dim=1).pow(1/2.).sum()
    #     return lasso
    def filter_lasso(weight):
        weight = tf.reduce_sum(weight, 0)
        weight = tf.reduce_sum(weight, 0)
        weight = tf.reduce_sum(weight, 1)
        weight = tf.sqrt(weight)
        lasso = tf.reduce_sum(weight)
        return lasso

    def channel_lasso(weight):
        weight = tf.reduce_sum(weight, 3)
        weight = tf.reduce_sum(weight, 2)
        weight = tf.reduce_sum(weight, 1)
        weight = tf.sqrt(weight)
        lasso = tf.reduce_sum(weight)
        return lasso

    def group_lasso(weight):
        weight_pow = tf.square(weight)
        g_lasso = 0.02*filter_lasso(weight_pow) + 0.022*channel_lasso(weight_pow)   #args.spasity_filter = 0.022, args.spasity_channel = 0.025
        return g_lasso

    # +group_lasso(conv11_filter)+group_lasso(conv12_filter)+group_lasso(conv13_filter)\
    total_loss= loss1\
                +group_lasso(conv21_filter)+group_lasso(conv22_filter)+group_lasso(conv23_filter)
    return total_loss
