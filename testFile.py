
import atCode as atrain
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = tf.reshape(x_train, shape=[x_train.shape[0], 784])
x_test = tf.reshape(x_test, shape=[x_test.shape[0], 784])

x_train = preprocessing.normalize(x_train, axis=1)
x_test = preprocessing.normalize(x_test, axis=1)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.astype(np.float64)
y_train = y_train.astype(np.float64)
x_test = x_test.astype(np.float64)
y_test = y_test.astype(np.float64)

#%%
'''
The test accuracy with in-silico BP and AT
'''
nn1 = atrain.AsymmetricTrain([784, 256, 256, 10])
nn1.initial_param()
nn1.param_dev_init()
nn1.p_dev_init()
nn1.take_test_data(x_test, y_test)
train_acc1, test_acc1 = nn1.inscilicoBP(x_train, y_train, epochs=300, batch_size=600, l_r=5e-5)

#%%
nn2 = atrain.AsymmetricTrain([784, 256, 256, 10])
nn2.initial_param()
nn2.param_dev_init()
nn2.p_dev_init()
nn2.take_test_data(x_test, y_test)
train_acc2, test_acc2 = nn2.atMethod(x_train, y_train, epochs=300, batch_size=600, l_r=5e-5, m_r1=0.5, m_r2=0.5)

#%%

epoc_list = np.arange(1,301,1)
plt.figure(figsize=[9,6])
plt.plot(epoc_list, test_acc1, label= "In silico BP")
plt.plot(epoc_list, test_acc2, label= "AT method")
plt.legend()
plt.show()


