#!/usr/bin/python3

from __future__ import division
import os
import sys
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, save_model
import rf_models
import time
import load_slice_IQ
import config
import get_simu_data
from tools import shuffleData
from statistics import mean, stdev

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(ROOT_DIR)
resDir = os.path.join(ROOT_DIR, 'res_out')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)

class LearningController(Callback):
    def __init__(self, num_epoch=0, lr = 0., learn_minute=0):
        self.num_epoch = num_epoch
        self.learn_second = learn_minute * 60
        self.lr = lr
        if self.learn_second > 0:
            print("Leraning rate is controled by time.")
        elif self.num_epoch > 0:
            print("Leraning rate is controled by epoch.")

    def on_train_begin(self, logs=None):
        if self.learn_second > 0:
            self.start_time = time.time()
        self.model.optimizer.lr = self.lr


    def on_epoch_end(self, epoch, logs=None):
        if self.learn_second > 0:
            current_time = time.time()
            if current_time - self.start_time > self.learn_second:
                self.model.stop_training = True
                print("Time is up.")
                return

            if current_time - self.start_time > self.learn_second / 2:
                self.model.optimizer.lr = self.lr * 0.1
            if current_time - self.start_time > self.learn_second * 3 / 4:
                self.model.optimizer.lr = self.lr * 0.01

        elif self.num_epoch > 0:
            if epoch >= self.num_epoch / 3:
                self.model.optimizer.lr = self.lr * 0.1
            if epoch >= self.num_epoch * 2 / 3:
                self.model.optimizer.lr = self.lr * 0.01

        print('lr:%.2e' % self.model.optimizer.lr.value())



def main(opts):
    # load data
    same_acc_list = []
    cross_acc_list = []
    target = os.path.basename(opts.trainData)
    outfile = os.path.join(resDir, 'multiruns_cross_day_{}_res.txt'.format(target))
    f = open(outfile, 'a+')
    print('\n\n##################### dataType: {}, slice_len: {}, window: {}, test time is: {}####################'.format(opts.dataType, opts.slice_len, opts.window, time.ctime()), file=f, flush=True)
    resLine = ''

    # setup params
    Batch_Size = 64
    Epoch_Num = 100
    lr = 0.1
    emb_size = 64
    idx_list = [0,100000,200000,300000,400000]
    #idx_list = [0,0,0]
    for idx in idx_list:
        dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, num_slice=opts.num_slice,
                                              slice_len=opts.slice_len,
                                              start_idx=idx, stride=opts.stride, mul_trans=opts.mul_trans,window=opts.window,
                                              dataType=opts.dataType)

        train_x, train_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
        saveModelPath = os.path.join(modelDir, '{}_model_{}_{}_{}_slices_{}_startIdx_{}_stride_{}_len_{}_STFT_{}'.format(opts.dataType, target, opts.modelType, opts.location, opts.num_slice, idx, opts.stride,opts.slice_len, opts.window))
        checkpointer = ModelCheckpoint(filepath=saveModelPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
        learning_controller = LearningController(num_epoch=Epoch_Num, lr=lr)
        callBackList = [checkpointer, earlyStopper]

        print('get the model and compile it...')
        inp_shape = (train_x.shape[1], train_x.shape[2])
        print('input shape: {}'.format(inp_shape))
        # pdb.set_trace()
        model = rf_models.create_model(opts.modelType, inp_shape, NUM_CLASS, emb_size, classification=True)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        train_y = to_categorical(train_y, NUM_CLASS)
        test_y = to_categorical(test_y, NUM_CLASS)
        train_x, train_y = shuffleData(train_x, train_y)
        print('fit the model with data...')
        start_time = time.time()
        model.fit(x=train_x, y=train_y,
               batch_size=Batch_Size,
               epochs=Epoch_Num,
               verbose=opts.verbose,
               callbacks=callBackList,
               validation_split=0.2,
               shuffle=True)
        end_time = time.time()
        duration = end_time - start_time
        print('test the trained model...')
        m = load_model(saveModelPath)
        score, acc = m.evaluate(test_x, test_y, batch_size=Batch_Size, verbose=1)
        same_acc_list.append(acc)
        m.save(saveModelPath + '_{:.2f}'.format(acc))
        print('test acc is: ', acc)

        resLine = resLine + 'dataset: {}, model: {}, location: {}, data size: {}, start_idx: {}, stride: {}\n'.format(target, opts.modelType, opts.location, opts.num_slice, idx, opts.stride)
        resLine = resLine + 'same acc is: {:f}, time last: {:f}\n\n'.format(acc, duration)


        print("start testing on cross day scenario...")

        dataOpts = load_slice_IQ.loadDataOpts(opts.testData, opts.location, num_slice=opts.num_slice, slice_len=opts.slice_len,
                                           start_idx = idx, stride = opts.stride, mul_trans = opts.mul_trans, window = opts.window, dataType=opts.dataType)

        dataOpts.num_slice = int(opts.num_slice * 0.2)
        X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
        y = to_categorical(y, NUM_CLASS)
        X, y = shuffleData(X, y)

        day2_score, day2_acc = m.evaluate(X, y, batch_size=Batch_Size, verbose=1)
        print('day2 test acc is: ', day2_acc)
        cross_acc_list.append(day2_acc)
        resLine = resLine + 'cross day acc is: {:f}, time last: {:f}\n\n'.format(day2_acc, duration)
    resLine = resLine + 'ave same acc is : {:f}, std: {:f}, ave cross acc is : {:f}, std: {:f}\n'.format(mean(same_acc_list), stdev(same_acc_list), mean(cross_acc_list), stdev(cross_acc_list))
    print(resLine, file=f, flush=True)
    print('all test done!')


class testOpts():
    
    def __init__(self, trainData, testData, location, modelType, num_slice, slice_len, start_idx, stride, window, dataType):
        self.input = trainData
        self.testData = testData
        self.modelType = modelType
        self.location = location
        self.verbose = 1
        self.trainData = trainData
        self.splitType = 'random'
        self.normalize = False
        self.dataSource = 'neu'
        self.num_slice = num_slice
        self.slice_len = slice_len
        self.start_idx = start_idx
        self.stride = stride
        self.window = window
        self.mul_trans = True
        self.dataType = dataType


if __name__ == "__main__":
    # opts = config.parse_args(sys.argv)
    source = ['our_day1']
    target = ['our_day2']
    data = list(zip(source, target))
    for s in [864]:
        for w in [64]:
            for m in ['homegrown']:     
                for p in data:
                    dataPath = '/home/haipeng/Documents/dataset/radio_dataset/' + p[0]
                    testPath = '/home/haipeng/Documents/dataset/radio_dataset/' + p[1]
                    opts = testOpts(trainData=dataPath, testData=testPath, location='symbols', modelType= m, num_slice= 5000, slice_len= 864, start_idx=0, stride = s, window=w, dataType='IQ')
                    main(opts)

