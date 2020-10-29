from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Concatenate, Input, Dense, Dropout, BatchNormalization,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import seaborn as sns

ROOT_DIR = Path(__file__).parent
DATASET_DIR = ROOT_DIR / 'dataset'

def load_dataset(dir):
    test_acc_x_file = open(dir/'test'/'Inertial Signals'/ 'body_acc_x_test.txt','r')
    test_acc_x = test_acc_x_file.read().splitlines()
    test_acc_x = [[float(x) for x in list(filter(None,x.split(' ') ))] for x in test_acc_x]
    test_acc_x = np.array(test_acc_x)
    test_acc_y_file = open(dir/'test'/'Inertial Signals'/ 'body_acc_y_test.txt','r')
    test_acc_y = test_acc_y_file.read().splitlines()
    test_acc_y = [[float(y) for y in list(filter(None,y.split(' ') ))] for y in test_acc_y]
    test_acc_y = np.array(test_acc_y)
    test_acc_z_file = open(dir/'test'/'Inertial Signals'/ 'body_acc_z_test.txt','r')
    test_acc_z = test_acc_z_file.read().splitlines()
    test_acc_z = [[float(z) for z in list(filter(None,z.split(' ') ))] for z in test_acc_z]
    test_acc_z = np.array(test_acc_z)
    test_acc = np.stack([test_acc_x,test_acc_y,test_acc_z],axis=-1)

    test_gyro_x_file = open(dir/'test'/'Inertial Signals'/ 'body_gyro_x_test.txt','r')
    test_gyro_x = test_gyro_x_file.read().splitlines()
    test_gyro_x = [[float(x) for x in list(filter(None,x.split(' ') ))] for x in test_gyro_x]
    test_gyro_x = np.array(test_gyro_x)
    test_gyro_y_file = open(dir/'test'/'Inertial Signals'/ 'body_gyro_y_test.txt','r')
    test_gyro_y = test_gyro_y_file.read().splitlines()
    test_gyro_y = [[float(y) for y in list(filter(None,y.split(' ') ))] for y in test_gyro_y]
    test_gyro_y = np.array(test_gyro_y)
    test_gyro_z_file = open(dir/'test'/'Inertial Signals'/ 'body_gyro_z_test.txt','r')
    test_gyro_z = test_gyro_z_file.read().splitlines()
    test_gyro_z = [[float(z) for z in list(filter(None,z.split(' ') ))] for z in test_gyro_z]
    test_gyro_z = np.array(test_gyro_z)
    test_gyro = np.stack([test_gyro_x,test_gyro_y,test_gyro_z],axis=-1)

    train_acc_x_file = open(dir/'train'/'Inertial Signals'/ 'body_acc_x_train.txt','r')
    train_acc_x = train_acc_x_file.read().splitlines()
    train_acc_x = [[float(x) for x in list(filter(None,x.split(' ') ))] for x in train_acc_x]
    train_acc_x = np.array(train_acc_x)
    train_acc_y_file = open(dir/'train'/'Inertial Signals'/ 'body_acc_y_train.txt','r')
    train_acc_y = train_acc_y_file.read().splitlines()
    train_acc_y = [[float(y) for y in list(filter(None,y.split(' ') ))] for y in train_acc_y]
    train_acc_y = np.array(train_acc_y)
    train_acc_z_file = open(dir/'train'/'Inertial Signals'/ 'body_acc_z_train.txt','r')
    train_acc_z = train_acc_z_file.read().splitlines()
    train_acc_z = [[float(z) for z in list(filter(None,z.split(' ') ))] for z in train_acc_z]
    train_acc_z = np.array(train_acc_z)
    train_acc = np.stack([train_acc_x,train_acc_y,train_acc_z],axis=-1)

    train_gyro_x_file = open(dir/'train'/'Inertial Signals'/ 'body_gyro_x_train.txt','r')
    train_gyro_x = train_gyro_x_file.read().splitlines()
    train_gyro_x = [[float(x) for x in list(filter(None,x.split(' ') ))] for x in train_gyro_x]
    train_gyro_x = np.array(train_gyro_x)
    train_gyro_y_file = open(dir/'train'/'Inertial Signals'/ 'body_gyro_y_train.txt','r')
    train_gyro_y = train_gyro_y_file.read().splitlines()
    train_gyro_y = [[float(y) for y in list(filter(None,y.split(' ') ))] for y in train_gyro_y]
    train_gyro_y = np.array(train_gyro_y)
    train_gyro_z_file = open(dir/'train'/'Inertial Signals'/ 'body_gyro_z_train.txt','r')
    train_gyro_z = train_gyro_z_file.read().splitlines()
    train_gyro_z = [[float(z) for z in list(filter(None,z.split(' ') ))] for z in train_gyro_z]
    train_gyro_z = np.array(train_gyro_z)
    train_gyro = np.stack([train_gyro_x,train_gyro_y,train_gyro_z],axis=-1)

    return train_acc,train_gyro,test_acc,test_gyro

def load_label(dir):

    label_dict = {
                'WALKING':1,
                'WALKING_UPSTAIRS':2,
                'WALKING_DOWNSTAIRS':3,
                'SITTING':4,
                'STANDING':5,
                'LAYING':6,
                }

    y_test_file = open(dir/'test' / 'y_test.txt','r')
    y_train_file = open(dir/'train' / 'y_train.txt','r')
    y_test = y_test_file.read().splitlines()
    y_train = y_train_file.read().splitlines()

    y_test = [int(x) for x in y_test]
    y_train = [int(x) for x in y_train]

    y_test = np.array(y_test)
    y_train = np.array(y_train)

    return y_train,y_test, label_dict

def mobility_score_label(label):

    new_label = []
    label_dict = {
                'WALKING':3,
                'WALKING_UPSTAIRS':5,
                'WALKING_DOWNSTAIRS':2,
                'SITTING':1,
                'STANDING':1,
                'LAYING':0,
                }
    for i in range(len(label)):
        if label[i] == 1:
            new_label.append(3.0)
        elif label[i] == 2:
            new_label.append(5.0)
        elif label[i] == 3:
            new_label.append(2.0)
        elif label[i] == 4:
            new_label.append(1.0)
        elif label[i] == 5:
            new_label.append(1.0)
        elif label[i] == 6:
            new_label.append(0.0)
    new_label = np.array(new_label)

    return new_label, label_dict
def four_class_label(label,onehot=True):
    new_label = []
    label_dict = {
                'WALKING':0,
                'SITTING':1,
                'STANDING':2,
                'LAYING':3,
                }
    for i in range(len(label)):
        if label[i] == 1 or label[i] == 2 or label[i] == 3:
            new_label.append(0)
        elif label[i] == 4:
            new_label.append(1)
        elif label[i] == 5:
            new_label.append(2)
        elif label[i] == 6:
            new_label.append(3)

    if onehot:
        new_label = np.array(new_label)
        new_label = tf.keras.utils.to_categorical(y=new_label,num_classes=4)
    return new_label, label_dict



def lstm_model():
    input1 = Input(shape=(128,3),name='acc_input')
    input2 = Input(shape=(128,3),name='gyro_input')


    left = LSTM(100,activation='relu',return_sequences=True,name='acc_LSTM')(input1)
    right = LSTM(100, activation='relu',return_sequences=True,name='gyro_LSTM')(input2)


    concat = Concatenate(axis=-1)([left,right])

    concat = LSTM(100, activation='relu',return_sequences=True,name='concatenated_LSTM_0')(concat)
    concat = LSTM(100, activation='relu',return_sequences=True,name='concatenated_LSTM_1')(concat)

    output1 = LSTM(100, activation='relu',return_sequences=True,name='four_class_LSTM')(concat)
    output2 = LSTM(100, activation='relu',return_sequences=True,name='mobility_score_LSTM')(concat)

    output1 = Flatten()(output1)
    output2 = Flatten()(output2)

    output1 = Dense(4,activation='softmax',name='four_class_output')(output1)
    output2 = Dense(1,name='mobility_score_output')(output2)

    model = tf.keras.Model(inputs=[input1,input2],outputs=[output1,output2])

    losses = {
        "four_class_output": "categorical_crossentropy",
        "mobility_score_output": "mse",
    }

    lossWeights = {"four_class_output": 1.0, "mobility_score_output": 1.0}


    metrics = {'four_class_output': 'accuracy', 'mobility_score_output': 'mse'}
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=losses, loss_weights=lossWeights,metrics=metrics)
    model.summary()
    return model
if __name__ == '__main__':



    train_acc,train_gyro,test_acc,test_gyro = load_dataset(DATASET_DIR)


    y_train, y_test, label_dict = load_label(DATASET_DIR)


    y_train_four_class,four_class_label_dict = four_class_label(y_train)
    y_test_four_class, _ = four_class_label(y_test)


    y_train_four_class_no_onehot,_ = four_class_label(y_train,onehot=False)
    y_test_four_class_no_onehot, _ = four_class_label(y_test,onehot=False)

    y_train_mobility_score,mobility_score_label_dict = mobility_score_label(y_train)
    y_test_mobility_score, _ = mobility_score_label(y_test)
    # print(list(y_test_mobility_score))
    # exit(1)
    print('Shape train_acc',train_acc.shape)
    print('Shape train_gyro',train_gyro.shape)
    print('Shape y_train',y_train.shape)

    print('Shape test_acc',test_acc.shape)
    print('Shape test_gyro',test_gyro.shape)
    print('Shape y_test',y_test.shape)



    model = lstm_model()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit(x=[train_acc,train_gyro],
                        y=[y_train_four_class,y_train_mobility_score],
                        steps_per_epoch=100,
                        validation_split=0.15,
                        epochs=1000,
                        callbacks=[es],
                        batch_size=16,
                        shuffle=True)

    print(history.history)

    plt.figure(figsize=(5,6))
    plt.subplot(211)
    plt.plot(history.history['four_class_output_accuracy'])
    plt.plot(history.history['val_four_class_output_accuracy'])
    plt.title('four_class_output_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.grid()

    plt.subplot(212)
    plt.plot(history.history['four_class_output_loss'])
    plt.plot(history.history['val_four_class_output_loss'])
    plt.title('four_class_output_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.grid()
    plt.savefig("four_class_learning_graph.png", dpi=300)
    plt.show()

    plt.figure(figsize=(5,6))
    plt.subplot(211)
    plt.plot(history.history['mobility_score_output_mse'])
    plt.plot(history.history['val_mobility_score_output_mse'])
    plt.title('mobility_score_output_mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.grid()

    # plt.subplot(212)
    # plt.plot(history.history['mobility_score_output_loss'])
    # plt.plot(history.history['val_mobility_score_output_loss'])
    # plt.title('mobility_score_output_loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'Validation'], loc='upper left')
    # plt.grid()
    # plt.savefig("mobility_score_learning_graph.png", dpi=300)
    plt.show()


    y_pred = model.predict([test_acc,test_gyro])

    predict_classifier = [np.argmax(x) for x in y_pred[0]]
    print(predict_classifier)
    print(y_test_four_class_no_onehot)

    print("MSE",mean_squared_error(y_test_mobility_score, y_pred[1]))

    cm = confusion_matrix(y_test_four_class_no_onehot, predict_classifier)
    sns.heatmap(cm, annot=True)
    plt.title("confusion matrix")
    plt.savefig("confusion_matrix_learning_graph.png", dpi=300)
    plt.show()

