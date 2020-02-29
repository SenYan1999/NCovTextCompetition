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
        train_data = BiSentDataset(args.raw_train_data, args.max_len)
        dev_data = BiSentDataset(args.raw_dev_data, args.max_len)

    # ESIM
    elif args.esim:
        train_data = EsimDataset(args.raw_train_data, args.max_len, args.min_occurance)
        dev_data = EsimDataset(args.raw_dev_data, args.max_len, args.min_occurance, word2idx=train_data.word2idx)
    
    # Bert-Esim
    elif args.bert_esim:
        train_data = BertEsimDataset(args.raw_train_data, args.max_len)
        dev_data = BertEsimDataset(args.raw_dev_data, args.max_len)

    else:
        assert(Exception('Please choose a model.'))

    save_pt(train_data, args.train_data)
    save_pt(dev_data, args.dev_data)

if __name__ == '__main__':
    main()