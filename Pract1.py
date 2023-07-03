# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 07:51:00 2023

@author: Aditi
"""

import tensorflow as tf
print('Matrix Multiplication Demo')
x = tf.constant([1,2,3,4,5,6], shape=[2,3])
print(x)

y = tf.constant([7,8,9,10,11,12], shape=[3,2])
print(y)

z = tf.matmul(x, y)
print('Product:',z)

e_matrix_A = tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32, name='matrixA')

print('Matrix A:\n{}\n\n'.format(e_matrix_A))
eigen_values_A, eigeen_vactor_A = tf.linalg.eigh(e_matrix_A)
print("Eigeen vactor:\n{}\n\n Eigen values:\n{}\n".format(eigeen_vactor_A,eigen_values_A))