import os

import numpy as np
import pandas as pd
from statistics import mean, stdev
import load_slice_IQ
import tools as mytools
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from utility import class_ranks
import time
RootDir = os.getcwd()
currentDir = os.path.dirname(__file__)
ResDir = os.path.join(currentDir, 'res_out')
os.makedirs(ResDir, exist_ok=True)

class tfOpts():
    def __init__(self, source_path, location, file_key='*.bin', num_slice=20000, start_ix=0, slice_len=288, stride=1):
        self.root_dir = source_path
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.sample = True


def ada_classifier_test():
    test_times = 5
    outfile = os.path.join(ResDir, 'ada_classifier_test_on_day2.txt')
    f = open(outfile, 'a+')
    print('\n\n#################### test time is: {} #########################'.format(time.ctime()), file=f)
    for arch in ["DF", "homegrown"]:
        for n in [1,5,10,20,40,60,80]:
            modelPath = '/home/haipeng/Documents/radio_fingerprinting/ADA/res_out/modelDir/ADA_{}_our_source_100000_target_{}'.format(arch, str(n))

            m = load_model(modelPath)
            m.summary()
            dataPath = "/home/haipeng/Documents/dataset/radio_dataset/"
            opts = tfOpts(source_path=dataPath + "test_dataset_2/", test_path= dataPath + "test_dataset_2/", location='symbols')
            dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location,  num_slice=opts.num_slice, slice_len=opts.slice_len, start_idx=0, sample=opts.sample)
            acc_list = []
            total_rank = []
            for i in range(test_times):
                x, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
                test_y = to_categorical(y, NUM_CLASS)
                test_x, test_y = mytools. shuffleData(x, test_y)

                score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
                acc_list.append(acc)
                print(acc)
                ranks = class_ranks(m, x, y, NUM_CLASS, preds=None)
                if i == 0:
                    total_rank = np.array(ranks)
                else:
                    total_rank += np.array(ranks)

            df = pd.DataFrame(np.rint(total_rank/test_times), index=None)
            name = os.path.basename(modelPath)
            df.to_csv("rank_result/rank_" + name + '.csv',header=False)

            print("acc: {}, std: {}".format(mean(acc_list), stdev(acc_list)))
            print("architecture: {}, n_shot: {}, ave acc: {}, std: {}".format(arch, n, mean(acc_list), stdev(acc_list)), file=f)
    f.close()


def model_test():
    test_times = 3
    comb = list()
    for s in [True, False]:
        for m in [True, False]:
            comb.append([s,m])
    outfile = os.path.join(ResDir, 'channel_imperfection.txt')
    f = open(outfile, 'a+')
    print('\n\n#################### NEU wired dataset, test time is: {} #########################'.format(time.ctime()), file=f)
    models = os.listdir('/home/haipeng/Documents/radio_fingerprinting/res_out/modelDir/neu_wired_models/')
    for m in models:
        modelPath = '/home/haipeng/Documents/radio_fingerprinting/res_out/modelDir/neu_wired_models/' + m
        model_name = os.path.basename(modelPath).split('_')
        sample = model_name[10]
        mul_trans = model_name[13]
        print(modelPath, sample, mul_trans)
        m = load_model(modelPath)
        m.summary()
        dataPath = "/home/haipeng/Documents/dataset/radio_dataset/"
        for c in comb:
            acc_list = list()
            for i in range(test_times):
                opts = tfOpts(source_path=dataPath + "neu_wired_day2/", location='after_equ')
                dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location,  num_slice=2000, slice_len=opts.slice_len, start_idx=0, sample= c[0], mul_trans = c[1])
                x, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
                test_y = to_categorical(y, NUM_CLASS)
                test_x, test_y = mytools. shuffleData(x, test_y)

                score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
                acc_list.append(acc)
                print("Source sample and mul_trans: {},{}, Target sample and mul_trans: {},{}. Acc: {}".format(sample, mul_trans, c[0], c[1], acc))
            print("Source sample and mul_trans: {},{}, Target sample and mul_trans: {},{}. Ave acc: {}, std: {}".format(sample, mul_trans, c[0], c[1], mean(acc_list), stdev(acc_list)), file=f)
    f.close()

def single_model_test():

    modelPath = '/home/haipeng/Documents/radio_fingerprinting/res_out/modelDir/best_model_neu_different_day3_DF_10000_sample_False_mul_trans_True_0_0.32'
    m = load_model(modelPath)
    m.summary()
    dataPath = "/home/haipeng/Documents/dataset/radio_dataset/neu_different_day4"
    opts = tfOpts(source_path=dataPath, location='after_equ')
    dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location,  num_slice = 10000, slice_len=opts.slice_len, start_idx = 0, sample = False, mul_trans = True)
    _, _, X, y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
    test_y = to_categorical(y, NUM_CLASS)
    test_x, test_y = mytools. shuffleData(X, test_y)

    score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print("Acc: {}".format(acc))


def rank_model_test():

    modelPath = '/home/haipeng/Documents/radio_fingerprinting/res_out/modelDir/best_model_neu_different_day3_DF_10000_sample_True_mul_trans_True_0_0.05'
    m = load_model(modelPath)
    m.summary()
    dataPath = "/home/haipeng/Documents/dataset/radio_dataset/neu_different_day3"
    opts = tfOpts(source_path=dataPath, location='after_equ')
    dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location, num_slice = 1000, slice_len=288, stride=576,  start_idx = 200000, sample = False, dataType='IQ')
    #test_time = 3
      
    X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
    test_y = to_categorical(y, NUM_CLASS)
    test_x, test_y = mytools. shuffleData(X, test_y)
    #for i in range(test_time):
          #_, _, X, y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
        #X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
        #test_y = to_categorical(y, NUM_CLASS)
        #test_x, test_y = mytools. shuffleData(X, test_y)

    score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print("Acc: {}".format(acc))

    ranks = class_ranks(m, X, y, NUM_CLASS, preds=None)
    #if i == 0:
    total_rank = np.array(ranks)
    #else:
    #    total_rank += np.array(ranks)
    #    print(total_rank.shape)

    df = pd.DataFrame(total_rank, index=None)
    name = os.path.basename(modelPath)
    df.to_csv("rank_result/sameday_rank_" + name + '.csv',header=False)

if __name__ == '__main__':
    rank_model_test()
    #ada_classifier_test()
    #model_test()
    #single_model_test()


