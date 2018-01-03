'''
 Author: XIE, Wentao
 Email: 11510010@mail.sustc.edu.cn
 Last modified: Dec. 27, 2017
'''
import random

import numpy as np
import pickle

import matplotlib.pyplot as plt

# Definition of functions and parameters
EPOCH = 100
ITERATION = 100
BATCH_SIZE = 100

LAMBDA = 0.0005  # Normalization factor
ITA = 0.4  # Initialized learning rate. Become 0.04 after 50 epochs

accuracy = []
loss = []

out = []
j = -1
e = 0


# RELU function: the none-linear function of the hidden layer 1, 2
def RELU(z_mtx):
    out = np.array(np.zeros(z_mtx.size)).reshape(z_mtx.shape)
    out.reshape(z_mtx.shape)
    for idx in range(0, out.shape[0]):
        out[idx] = max(z_mtx[idx], 0)
    return out


# Softmax function: the none-linear function for the output layer
def soft_max(z_mtx):
    sum = 0.0
    sigma_vct = []
    for z in z_mtx:
        sum += np.exp(z)
    for z in z_mtx:
        sigma_vct.append(np.exp(z) / sum)
    return np.array(sigma_vct)


# Partial derivative of RELU function
def diff_RELU(z_mtx):
    v = np.array(np.zeros((z_mtx.size)))
    for j in range(0, z_mtx.size):
        if z_mtx[j] > 0:
            v[j] = 1
        else:
            v[j] = 0
    return v


# Read all data from .pkl
(train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),
                                                                     encoding='latin1')

'''
1. Data pre-processing.
   Normalize all the pixels to [0, 1) by dividing by 256.
'''
train_images = train_images.astype(np.float64)
for i in range(0, train_images.shape[0]):
    for j in range(0, train_images.shape[1]):
        train_images[i][j] = train_images[i][j] / 256

test_images = test_images.astype(np.float64)
for i in range(0, test_images.shape[0]):
    for j in range(0, test_images.shape[1]):
        test_images[i][j] = test_images[i][j] / 256

'''
2. NN initialization.
   Initialize the weight matrix using Xavier method, 
   initialize the constant vector to 0.
'''
w1_mtx = np.array(np.zeros((300, 784)), np.float64)
w2_mtx = np.array(np.zeros((100, 300)), np.float64)
w3_mtx = np.array(np.zeros((10, 100)), np.float64)

for i in range(0, w1_mtx.shape[0]):
    for j in range(0, w1_mtx.shape[1]):
        w1_mtx[i][j] = random.uniform(-np.sqrt(6) / np.sqrt(w1_mtx.shape[0] + w1_mtx.shape[1]),
                                      np.sqrt(6) / np.sqrt(w1_mtx.shape[0] + w1_mtx.shape[1]))

for i in range(0, w2_mtx.shape[0]):
    for j in range(0, w2_mtx.shape[1]):
        w2_mtx[i][j] = random.uniform(-np.sqrt(6) / np.sqrt(w2_mtx.shape[0] + w2_mtx.shape[1]),
                                      np.sqrt(6) / np.sqrt(w2_mtx.shape[0] + w2_mtx.shape[1]))

for i in range(0, w3_mtx.shape[0]):
    for j in range(0, w3_mtx.shape[1]):
        w3_mtx[i][j] = random.uniform(-np.sqrt(6) / np.sqrt(w3_mtx.shape[0] + w3_mtx.shape[1]),
                                      np.sqrt(6) / np.sqrt(w3_mtx.shape[0] + w3_mtx.shape[1]))

b1_mtx = np.array(np.zeros(300), np.float64)
b2_mtx = np.array(np.zeros(100), np.float64)
b3_mtx = np.array(np.zeros(10), np.float64)

'''
3. Training the NN.
   The training is done in 100 epochs. In each epoch, the entire data set
   is traversed, and the whole data set is divided into 100 batch which is 
   of the size of 100 training images. 
   The training process in each batch is as follow: 
       1. Loading the image data;
       2. Forwarding the data in the NN;
       3. Computing the loss and back propagation;
       4. Gradient decreasing.
'''
for count in range(0, EPOCH):

    for itr in range(0, ITERATION):
        if count == EPOCH / 2:
            ITA = ITA / 10

        delta_w1 = np.array(np.zeros((300, 784)))
        delta_w2 = np.array(np.zeros((100, 300)))
        delta_w3 = np.array(np.zeros((10, 100)))

        delta_b1 = np.array(np.zeros(300))
        delta_b2 = np.array(np.zeros(100))
        delta_b3 = np.array(np.zeros(10))

        for idx in range(0, BATCH_SIZE):
            # Forward propagation
            input = train_images[itr * BATCH_SIZE + idx]
            h1_in = np.matmul(w1_mtx, input) + b1_mtx
            h1_out = RELU(h1_in)
            h2_in = np.matmul(w2_mtx, h1_out) + b2_mtx
            h2_out = RELU(h2_in)
            out_in = np.matmul(w3_mtx, h2_out) + b3_mtx
            out = soft_max(out_in)

            # Back propagation
            j = train_labels[itr * BATCH_SIZE + idx]
            l = np.array(np.zeros(10))
            l[j] = 1

            delta_out = out - l
            delta_h2 = np.matmul(w3_mtx.T, delta_out) * diff_RELU(h2_out)
            delta_h1 = np.matmul(w2_mtx.T, delta_h2) * diff_RELU(h1_out)

            delta_w3 += (np.matmul(delta_out.reshape((10, 1)), h2_out.reshape((1, 100))))
            delta_w2 += (np.matmul(delta_h2.reshape((100, 1)), h1_out.reshape((1, 300))))
            delta_w1 += (np.matmul(delta_h1.reshape((300, 1)), train_images[itr * BATCH_SIZE + idx].reshape((1, 784))))

            delta_b3 += delta_out
            delta_b2 += delta_h2
            delta_b1 += delta_h1

            if itr == ITERATION - 1:
                e += -np.log(out[j])

        # Gradient update
        w1_mtx -= (delta_w1 / BATCH_SIZE + LAMBDA * w1_mtx) * ITA
        w2_mtx -= (delta_w2 / BATCH_SIZE + LAMBDA * w2_mtx) * ITA
        w3_mtx -= (delta_w3 / BATCH_SIZE + LAMBDA * w3_mtx) * ITA
        b1_mtx -= delta_b1 / BATCH_SIZE * ITA
        b2_mtx -= delta_b2 / BATCH_SIZE * ITA
        b3_mtx -= delta_b3 / BATCH_SIZE * ITA
    loss.append(e / BATCH_SIZE)

    # Testing the NN using other 1000 image data
    cnt = 0
    for i in range(0, test_labels.size):

        input = test_images[i]
        h1_in = np.matmul(w1_mtx, input) + b1_mtx
        h1_out = RELU(h1_in)
        h2_in = np.matmul(w2_mtx, h1_out) + b2_mtx
        h2_out = RELU(h2_in)
        out_in = np.matmul(w3_mtx, h2_out) + b3_mtx
        out = soft_max(out_in)
        m = max(out)
        result = -1
        for k in range(0, out.size):
            if out[k] == m:
                result = k
                break
        truth = test_labels[i]
        if result == truth:
            cnt = cnt + 1
    accuracy.append(float(cnt) / float(test_labels.size))

t = []
for itm in loss:
    t.append(round(itm, 4))


plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(accuracy)
plt.xlabel('#iteration')
plt.ylabel('Accuracy')
plt.grid()
plt.tight_layout()
plt.savefig('figure.pdf', dbi=300)
