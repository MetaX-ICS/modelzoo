import os
import numpy as np

from .psepostprocess import get_pse_post, get_dbnet_post
from .mmdbpostprocess import  get_mmdbnet_post
from .tools import preprocess_test

import logging

def get_results(type, outputs, batch_size, params, content):
    return eval(type)(outputs, batch_size, params, content)

def write_result_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)


def comute_Fm(predns, params, iters):

    image_names = preprocess_test(params['image_T4'])[:iters]

    for i in range(iters):
        image_name = image_names[i][0].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, predns[i], "./outputs")

    cmd = 'zip -q -j %s %s/* && rm -rf %s' % ('submit_ic15.zip', "outputs", "outputs")
    # print(cmd)
    response = os.system(cmd)

    cmd1 = "bash demo/eval/eval_ic15.sh ../../../submit_ic15.zip"
    # print(cmd1)
    response1 = os.system(cmd1)

def comute_crnn_acc(preds, iters):
    n_correct = np.array(preds).sum()
    n_total = iters
    accuracy = n_correct / float(n_total)
    print('accuracy: {:.4f}%'.format(accuracy*100))
    # logging.info('accuracy: {:.4f}%'.format(accuracy*100))
    return accuracy

def comute_pdocr_acc(preds, params):
    converter = strLabelConverter(params["label_list"])
    contents = []
    with open(params['image_T4']) as f:
        for line in f.readlines():
            contents.append(line.strip().split())

    n_total = len(preds)
    n_correct = 0
    for i in range(n_total):
        sim_pred = converter.decode(preds[i], preds[i].size, raw=False)
        if sim_pred == contents[i][1]:
            n_correct += 1

    accuracy = n_correct / float(n_total + 1e-6)
    print('accuracy: {:.4f}%'.format(accuracy*100))
    # logging.info('accuracy: {:.4f}%'.format(accuracy*100))
    return accuracy

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_file=None, ignore_case=False):
        self._ignore_case = ignore_case
        if alphabet_file is None:
            alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        else:
            alphabet = []
            with open(alphabet_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip("\n").strip("\r\n")
                    alphabet.append(line)
            alphabet = "".join(alphabet)
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        # if length.numel() == 1:
        # length = length[0]
        # assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),length)
        assert t.size == length, "text with length: {} does not match declared length: {}".format(t.size,length)
        
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)

def get_crnn_post(outputs, batch_size, params, content):

    converter = strLabelConverter()
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[0][0],outputs_size_list[0][1]) for i in range(len(outputs))]

    n_correct = []
    for i in range(batch_size):
        output = outputs[0][i]

        preds = output.argmax(1)
        # preds = preds.transpose(1, 0).reshape(-1)
        preds_size = preds.size

        # raw_pred = converter.decode(preds, preds_size, raw=True)
        sim_pred = converter.decode(preds, preds_size, raw=False)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))

        # target = converter.encode(label)
        if sim_pred == content[i][1].lower():
            n_correct.append(1)
        else:
            n_correct.append(0)

        # npreds.append(sim_pred)

    return n_correct

def get_chv3_post(outputs, batch_size, params, content):

    # converter = strLabelConverter(params["label_list"])
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[0][0],outputs_size_list[0][1]) for i in range(len(outputs))]

    npreds = []
    for i in range(batch_size):
        output = outputs[0][i]

        preds = output.argmax(1)
        # preds = preds.transpose(1, 0).reshape(-1)
        # preds_size = preds.size

        # raw_pred = converter.decode(preds, preds_size, raw=True)
        # sim_pred = converter.decode(preds, preds_size, raw=False)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))

        # # target = converter.encode(label)
        # if sim_pred == content[i][1].lower():
        #     n_correct.append(1)
        # else:
        #     n_correct.append(0)

        npreds.append(preds)

    return npreds
