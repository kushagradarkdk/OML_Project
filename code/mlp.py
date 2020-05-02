from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from adam import Adam
from aadam import AAdam
from sgd import SGD
from asgd import ASGD
from adagrad import Adagrad
from aadagrad import AAdagrad
import numpy as np 
import pandas as pd


results_accuracy = []
result_accuracy= []
results_loss = []
result_loss = []
test_accuracy_results = []
test_loss_results = []
x = 0

#Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#Data augmentation : Reshape and normalize
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#Change the label to Categorical data
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Build the model
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#Model summary
network.summary()
optimizers= [Adagrad(),AAdagrad(),SGD(),ASGD(), Adam(amsgrad = True), AAdam(amsgrad = True),Adam(amsgrad = False), AAdam(amsgrad = False) ]

for opt in optimizers:
    print(opt)
    network.compile(optimizer= opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    if x == 0:
        network.save_weights('initial_weights_mlp.h5')
        x = x+1
    network.load_weights('initial_weights_mlp.h5')
    initial_weights_mlp = network.get_weights()
    result_accuracy = []
    result_loss = []
    test_loss = []
    test_acc = []
    for i in range (20):
        network.load_weights('initial_weights_mlp.h5')
        result_accuracy_e = []
        result_loss_e = []
        test_acc_e = []
        test_loss_e = []
        for j in range (10):
            history = network.fit(train_images, train_labels, epochs=1, batch_size=128,verbose=0)
            if j % 2 == 0:
                test_loss_j, test_acc_j = network.evaluate(test_images, test_labels)
                test_acc_e.append(test_acc_j)
                test_loss_e.append(test_loss_j)
            result_accuracy_e.append(history.history['accuracy'][0])
            result_loss_e.append(history.history['loss'][0])
        test_loss.append(test_loss_e)
        test_acc.append(test_acc_e)
        result_accuracy.append(result_accuracy_e)
        result_loss.append(result_loss_e)
    print("----- NEW OPTIMIZER -----")
    print(opt)
    print(np.mean(result_accuracy,axis=0))
    print(np.mean(result_loss,axis=0))
    print(np.mean(test_acc,axis=0))
    print(np.mean(test_loss,axis=0))
    results_accuracy.append(np.mean(result_accuracy,axis=0))
    results_loss.append(np.mean(result_loss,axis=0))
    test_accuracy_results.append(np.mean(test_acc,axis=0))
    test_loss_results.append(np.mean(test_loss,axis=0))

df = pd.DataFrame(results_accuracy)
df.to_csv("resultsECML/acc_train_mlp.csv")
df = pd.DataFrame(results_loss)
df.to_csv("resultsECML/loss_train_mlp.csv")
df = pd.DataFrame(test_accuracy_results)
df.to_csv("resultsECML/acc_test_mlp.csv")
df = pd.DataFrame(test_loss_results)
df.to_csv("resultsECML/loss_test_mlp.csv")


