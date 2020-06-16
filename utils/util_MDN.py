import numpy as np
import tensorflow as tf


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """ 2D normal distribution
    input
    - x,mu: input vectors
    - s1,s2: standard deviances over x1 and x2
    - rho: correlation coefficient in x1-x2 plane
    """
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2.0*tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
    negRho = 1-tf.square(rho)
    result = tf.exp(tf.div(-1.0*z,2.0*negRho))
    denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
    px1x2 = tf.div(result, denom)
    return px1x2


 
