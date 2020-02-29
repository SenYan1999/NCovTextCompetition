import argparse

parser = argparse.ArgumentParser()

# model type
parser.add_argument('--bert', action='store_true')
parser.add_argument('--esim', action='store_true')
parser.add_argument('--bert_esim', action='store_true')

# data preprocess
parser.add_argument('--raw_train_data', type=str, default='data/train.csv')
parser.add_argument('--raw_dev_data', type=str, default='data/dev.csv')
parser.add_argument('--train_data', type=str, default='data/train.pt')
parser.add_argument('--dev_data', type=str, default='data/dev.pt')
parser.add_argument('--max_len', type=int, default='100')
parser.add_argument('--min_occurance', type=int, default='1')

# model
parser.add_argument('--bert_type', type=str, default='bert-base-chinese')
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--d_hidden', type=int, default=256)
parser.add_argument('--drop_out', type=float, default=0.2)

# train
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)

# save & log
parser.add_argument('--log_file', type=str, default='log/log.log')
parser.add_argument('--model_dir', type=str, default='save_model/model.pt')

# evaluate
parser.add_argument('--raw_test_data', type=str, default='/data/test.csv')
parser.add_argument('--out_file', type=str, default='result.csv')
