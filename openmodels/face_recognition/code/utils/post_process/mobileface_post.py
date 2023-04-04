import numpy as np
from itertools import product
from math import ceil

import imageio
import scipy.io


def parseList(pairs):
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        # p = p.split('\t')
        if len(p) == 3:
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            fold = i // 600
            flag = -1
        folds.append(fold)
        flags.append(flag)
    # print(nameLs)
    return folds, flags


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(result):
    ACCs = np.zeros(10)
    # result = scipy.io.loadmat(root)
    for i in range(10):
        fold = np.array(result['fold']).reshape((1,-1))
        flags = np.array(result['flag']).reshape((1,-1))
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
    #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    # print('--------')
    # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs

def comute_mobface_acc(predns, folds_flags):
    folds, flags = folds_flags
    featureLs = np.concatenate([fs[0] for fs in predns], 0)
    featureRs = np.concatenate([fs[1] for fs in predns], 0)
    result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
    # save tmp_result
    # scipy.io.savemat('./data/tmp_result.mat', result)
    accs = evaluation_10_fold(result)

    final_acc = np.mean(accs) #* 100
    print('ave: {:.4f}'.format(final_acc))
    return final_acc


def get_mobileface_post(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0]) for i in range(len(outputs))]

    features = []
    for idx in range(batch_size):
        res = outputs[0]
        featureL = np.concatenate((res[4*idx+0:4*idx+1], res[4*idx+1:4*idx+2]), 1)
        featureR = np.concatenate((res[4*idx+2:4*idx+3], res[4*idx+3:4*idx+4]), 1)

        features.append([featureL, featureR])

    return features