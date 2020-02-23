import torch
import csv

from transformers import BertTokenizer
from utils import *

args = parser.parse_args()

def convert_source_words(source, bert_type):
    data = []
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    with open(source, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            sent1, sent2, label = line[1], line[2], line[3]
            sent1, sent2 = tokenizer.tokenize(sent1), tokenizer.tokenize(sent2)
            try:
                label = int(label)
            except:
                continue
            data.append((sent1, sent2, int(label)))

    logger.info("Get {} sentences from file {}".format(len(data), source))

    return data

def main():
    # BERT
    if args.bert:
        train_data = convert_source_words(args.raw_train_data, args.bert_type)
        train_dataset = BiSentDataset(train_data, args.max_len)

        dev_data = convert_source_words(args.raw_dev_data, args.bert_type)
        dev_dataset = BiSentDataset(dev_data, args.max_len)

        save_pt(train_dataset, args.train_data)
        save_pt(dev_dataset, args.dev_data)

    # ESIM
    elif args.esim:
        train_data = EsimDataset(args.raw_train_data, args.max_len, args.min_occurance)
        dev_data = EsimDataset(args.raw_dev_data, args.max_len, args.min_occurance, word2idx=train_data.word2idx)
        save_pt(train_data, args.train_data)
        save_pt(dev_data, args.dev_data)
    
    else:
        assert(Exception('Please choose a model.'))

if __name__ == '__main__':
    main()