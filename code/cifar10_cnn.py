from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from adam import Adam
from aadam import AAdam
import numpy as np
import pandas as pd
from kdaAdam import KDAAdam

#Parameters
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
results_accuracy = []
result_accuracy= []
results_loss = []
result_loss = []
test_accuracy_results = []
test_loss_results = []
x=0

# split data between train and test sets:
print(keras.__version__)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

#Model Structure
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Print Model Summary
model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


optimizers= [Adam(learning_rate=0.0001,amsgrad = False), AAdam(learning_rate=0.0001,amsgrad = False), Adam(learning_rate=0.0001,amsgrad = True), AAdam(learning_rate=0.0001,amsgrad = True)]
for opt in optimizers:
    result_accuracy = []
    result_loss = []
    test_loss = []
    test_acc = []
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    if x==0:
        model.save_weights('initial_weights_cifar.h5')
        x = x+1
    model.load_weights('initial_weights_cifar.h5')
    initial_weights = model.get_weights()
    for i in range (2):
        result_accuracy_e = []
        result_loss_e = []
        test_acc_e = []
        test_loss_e = []
        model.set_weights(initial_weights)
        for j in range (20):
            history = model.fit(x_train, y_train,batch_size=batch_size,epochs=1,verbose=1)
            if j % 2 == 0 :
                test_loss_j, test_acc_j = model.evaluate(x_test, y_test)
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
    test_accuracy_results.append(np.mean(test_loss,axis=0))


df = pd.DataFrame(results_accuracy)
df.to_csv("results/cifar_acc_train_cnn.csv")
df = pd.DataFrame(results_loss)
df.to_csv("results/cifar_loss_train_cnn.csv")
df = pd.DataFrame(test_accuracy_results)
df.to_csv("results/cifar_acc_test_cnn.csv")
df = pd.DataFrame(test_accuracy_results)
df.to_csv("results/cifar_loss_test_cnn.csv")



