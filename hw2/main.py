import cv2
from pathlib import Path
import os
import random
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns

DATASET_DIR = Path('Data')

def data_loader(dataset_dir,testset=0.2):

    '''Initialize Container'''
    data = []
    label = []

    '''Loop through all the files in dataset directory'''
    for image_file in os.listdir(dataset_dir):
        image_path = dataset_dir / image_file

        '''Read Image'''
        image = cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)

        '''Find Lable for the image'''
        if 'y' in image_file.lower():
            label.append(1)
        elif 'n' in image_file.lower():
            label.append(0)

        '''Resizing'''
        data.append(image)

    '''Ramdomly create indexes with test train split'''
    testset_indexes = random.sample([x for x in range(len(data))], int(len(data) * testset))
    trainset_indexes = [x for x in range(len(data)) if x not in testset_indexes]

    '''Split lable and data'''
    test_label = [label[x] for x in testset_indexes]
    train_label = [label[x] for x in trainset_indexes]

    test_data = [data[x] for x in testset_indexes]
    train_data = [data[x] for x in trainset_indexes]

    return train_data,train_label,test_data,test_label

def transformer(data, label):

    '''Resize into (240,240)'''
    for i in range(len(data)):
        data[i] = cv2.resize(data[i], dsize=(240, 240))

    label = utils.to_categorical(label,num_classes=2,dtype='uint8')

    return data, label

def preprocessor(data):
    '''Convert to Numpy array'''
    data = np.array(data)

    '''Normalizing it by dividing into 255'''
    data = data / 255

    '''finalize input shape'''
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
    print(data.shape)
    return data

def ConvNet1(learning_rate=0.001):

    model = Sequential()

    model.add(layers.Conv2D(filters=64,
                            kernel_size=(7,7),
                            strides=(1,1),
                            activation='relu',
                            padding='same',
                            input_shape=(240,240,1)))

    model.add(layers.MaxPool2D(pool_size=(4,4)))

    model.add(layers.Conv2D(filters=32,
                            kernel_size=(7,7),
                            strides=(1,1),
                            activation='relu',
                            padding='same'))

    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(2,activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model

def MyConv():
    model = Sequential()

    model.add(layers.Conv2D(filters=64,
                            kernel_size=(4,4),
                            strides=(1,1),
                            activation='relu',
                            padding='same'))

    model.add(layers.MaxPool2D(pool_size=(4, 4), padding='same'))

    model.add(layers.Dropout(.5))

    model.add(layers.Conv2D(filters=32,
                            kernel_size=(4,4),
                            strides=(1,1),
                            activation='relu',
                            padding='same'))

    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(layers.Dropout(.25))

    return model

def ConvNet2(learning_rate=0.001):

    model = Sequential()

    model.add(layers.InputLayer(input_shape=(240,240,1)))

    model.add(MyConv())
    model.add(MyConv())
    model.add(MyConv())
    model.add(MyConv())

    model.add(layers.Flatten())

    model.add(layers.Dense(2,activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model

def plot_train_and_val(history,batch_size,lr,epochs,model_name='default'):

    subtitle = ' B: '+ str(batch_size) + ' LR: ' + str(lr) + ' E: ' + str(epochs)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy' + subtitle)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.savefig(model_name+'Best_Model_ACC.png', dpi=300)
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss' + subtitle)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.savefig(model_name+'Best_Model_Loss.png', dpi=300)
    plt.show()
def grid_search(build_fn,model_name='default'):
    model = KerasClassifier(build_fn=build_fn, verbose=1)

    batch_sizes = [8, 16,32]
    epochs = [5,20,40,120,480,960]
    learning_rates = [0.01,0.001,0.0001,0.00001,0.000001]

    param_grid = dict(batch_size=batch_sizes, epochs=epochs, learning_rate=learning_rates)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid.fit(train_data, train_label)


    for i, lr in enumerate(learning_rates):

        grid_data_frame = pd.DataFrame(grid.cv_results_)
        indexes_to_drop = []
        for j in range(len(grid_data_frame)):
            if float(grid_data_frame['params'][j]['learning_rate']) != lr:
                indexes_to_drop.append(j)
        df = grid_data_frame.drop(indexes_to_drop)

        pvt = pd.pivot_table(df, values='mean_test_score', index='param_epochs', columns='param_batch_size')

        sns.heatmap(pvt, annot=True)
        plt.title('Learning Rate ' + str(lr))
        plt.savefig(model_name+'_'+str(lr)+'_'+'Grid_Search_Result.png', dpi=300)
        plt.show()
    return grid

def train_and_test_best_model(model,grid_search_result,model_name='default',batch_size=32,epoch=25,learning_rate=0.001):
    if grid_search_result:
        best_batch_size = grid_search_result.best_params_['batch_size']
        best_epoch = grid_search_result.best_params_['epochs']
        best_learning_rate = grid_search_result.best_params_['learning_rate']
    else:
        best_batch_size = batch_size
        best_epoch = epoch
        best_learning_rate = learning_rate


    best_model = model(best_learning_rate)

    history = best_model.fit(x=train_data,
                        y=train_label,
                        batch_size=best_batch_size,
                        epochs=best_epoch,
                        validation_data=(test_data,test_label))

    score = best_model.evaluate(x=test_data,
                           y=test_label,
                           batch_size=best_batch_size)

    plot_train_and_val(history,best_batch_size,best_learning_rate,best_epoch,model_name)

    samples = random.sample(list(test_data),10)
    predict_result = best_model.predict_classes(np.array(samples))


    fig, axs = plt.subplots(2,5)

    for i in range(len(samples)):
        result = predict_result[i]
        sample_image = samples[i]

        axs[int(i / 5)][i % 5].imshow(sample_image,'gray')
        axs[int(i / 5)][i % 5].set_title("Label" + str(result))


    plt.savefig(model_name+'Sample_Test_Result.png', dpi=300)
    plt.show()
    return score

if __name__ == '__main__':

    '''Loading dataset and split'''
    train_data,train_label,test_data,test_label = data_loader(DATASET_DIR,testset=0.2)

    '''Resizing into (240,240)'''
    train_data,train_label = transformer(train_data,train_label)
    test_data,test_label = transformer(test_data,test_label)

    '''Normalize'''
    train_data,test_data = (preprocessor(train_data) , preprocessor(test_data))

    '''Traning model and evaluate'''
    train_and_test_best_model(model=ConvNet2,
                              grid_search_result=None,
                              model_name='BestConv2',
                              batch_size=128,
                              epoch=1600,
                              learning_rate=0.00005)

    train_and_test_best_model(model=ConvNet1,
                              grid_search_result=None,
                              model_name='BestConv1',
                              batch_size=24,
                              epoch=800,
                              learning_rate=0.000001)



    '''Grid Searching'''
    result = grid_search(build_fn=ConvNet1,model_name='ConvNet1')

    best_batch_size = result.best_params_['batch_size']
    best_epoch = result.best_params_['epochs']
    best_learning_rate = result.best_params_['learning_rate']

    print("Grid Search Result")
    print("Best Batch Size",best_batch_size)
    print("Best Eopch",best_epoch)
    print("Best Learning Rate",best_learning_rate)

    score = train_and_test_best_model(model=ConvNet1,grid_search_result=result,model_name='ConvNet1')

    print(score)





    result = grid_search(build_fn=ConvNet2,model_name='ConvNet2')

    best_batch_size = result.best_params_['batch_size']
    best_epoch = result.best_params_['epochs']
    best_learning_rate = result.best_params_['learning_rate']

    print("Grid Search Result")
    print("Best Batch Size",best_batch_size)
    print("Best Eopch",best_epoch)
    print("Best Learning Rate",best_learning_rate)

    score = train_and_test_best_model(model=ConvNet2,grid_search_result=result,model_name='ConvNet2')

    print(score)






















































