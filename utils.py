import torch
import pickle
import logging
import csv
import jieba

from collections import Counter
from logging import handlers
from torch.utils.data import Dataset
from transformers import BertTokenizer
from args import parser

args = parser.parse_args()

def save_pt(source, target):
    with open(target, 'wb') as f:
        pickle.dump(source, f)

def load_pt(file):
    with open(file, 'rb') as f:
        result = pickle.load(f)
    return result

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger
logger = init_logger(filename=args.log_file)

class BiSentDataset(Dataset):
    def __init__(self, data, max_len):
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_type)
        self.sents, self.labels, self.input_idx = self.convert_data(data, max_len)

    def convert_data(self, data, max_len):
        sents, labels, input_idx = [], [], []
        count = 0
        for line in data:
            sent1, sent2, label = line[0], line[1], line[2]
            idx_sent = [0] * (len(sent1) + 2) + [1] * (len(sent2) + 1)

            sent = ['[CLS]'] + sent1 + ['[SEP]'] + sent2 + ['[SEP]']
            if len(sent) < max_len:
                sent += ['[PAD]' for _ in range(max_len - len(sent))]
                idx_sent += [0 for _ in range(max_len - len(idx_sent))]
            else:
                sent = sent[:max_len]
                idx_sent = idx_sent[:max_len]
            sent = self.tokenizer.convert_tokens_to_ids(sent)
            assert len(idx_sent) == len(sent)
            
            sents.append(sent)
            labels.append(label)
            input_idx.append(idx_sent)

        sents, labels, input_idx = torch.LongTensor(sents), torch.LongTensor(labels), torch.LongTensor(input_idx)
        return sents, labels, input_idx
    
    def __getitem__(self, index):
        return (self.sents[index], self.labels[index], self.input_idx[index])

    def __len__(self):
        return self.sents.shape[0]

class EsimDataset(Dataset):
    def __init__(self, source_file, max_len, min_occurance, word2idx=None):
        data, word_counter = self.extract_data_from_source(source_file)
        self.word2idx = self.build_word2idx(word_counter, min_occurance) if word2idx == None else word2idx

        self.sent1, self.sent1_len, self.sent2, self.sent2_len, self.labels = \
            self.convert_data(data, self.word2idx, max_len)
    
    def extract_data_from_source(self, source):
        word_counter = Counter()
        data = []

        with open(source, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                sent1, sent2, label = line[1], line[2], line[3]

                try:
                    label = int(label)
                except:
                    continue
                sent1, sent2 = list(jieba.cut(sent1)), list(jieba.cut(sent2))

                word_counter.update(sent1 + sent2)
                data.append((sent1, sent2, label))
        
        return data, word_counter
    
    def build_word2idx(self, word_counter, min_occurance):
        word2idx = {}
        word2idx['[PAD]'] = 0
        word2idx['[UNK]'] = 1

        for idx, word in enumerate(word_counter, 2):
            if word_counter[word] > min_occurance:
                word2idx[word] = idx
        
        return word2idx

    def convert_data(self, data, word2idx, max_len):
        sent1s, sent1_lens, sent2s, sent2_lens, labels = [], [], [], [], []
        count = 0
        for line in data:
            sent1, sent2, label = line[0], line[1], line[2]
            sent1_len, sent2_len = len(sent1), len(sent2)

            if len(sent1) < max_len:
                sent1 += ['[PAD]' for _ in range(max_len - len(sent1))]
            else:
                sent1 = sent1[:max_len]
            if len(sent2) < max_len:
                sent2 += ['[PAD]' for _ in range(max_len - len(sent2))]
            else:
                sent2 = sent2[:max_len]

            sent1 = [word2idx.get(word, word2idx['[UNK]']) for word in sent1]
            sent2 = [word2idx.get(word, word2idx['[UNK]']) for word in sent2]
            
            sent1s.append(sent1)
            sent2s.append(sent2)
            sent1_lens.append(sent1_len)
            sent2_lens.append(sent2_len)
            labels.append(label)

        sent1s, sent1_lens, sent2s, sent2_lens, labels = \
            torch.LongTensor(sent1s), torch.LongTensor(sent1_lens), \
                torch.LongTensor(sent2s), torch.LongTensor(sent2_lens), \
                    torch.LongTensor(labels)
        return sent1s, sent1_lens, sent2s, sent2_lens, labels
    
    def __getitem__(self, index):
        return (self.sent1[index], self.sent1_len[index], self.sent2[index], \
            self.sent2_len[index], self.labels[index])

    def __len__(self):
        return self.sent1.shape[0]

if __name__ == "__main__":
    dataset = EsimDataset(args.raw_train_data, args.max_len, args.min_occurance)
    print(dataset[1])
    print(len(dataset))