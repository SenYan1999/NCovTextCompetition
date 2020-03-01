import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BaseBertModel, ESIM, BertESIM, TextCNN
from args import parser
from utils import *
from math import log2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_epoch(model, optimizer, scheluder, dataloader, epoch):
    logger.info('Epoch %2d: Training...' % epoch)
    model.train()
    loss_all = []
    loss_interval = []
    acc_all = []
    acc_interval = []

    pbar = tqdm(dataloader)

    for batch in pbar:
        optimizer.zero_grad()

        # forward
        if args.bert:
            idx, x, y, input_id = map(lambda x: x.to(device), batch)
            pred = model(x, input_id)
        elif args.esim:
            sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
            pred = model(sent1, sent1_len, sent2, sent2_len)
        elif args.bert_esim:
            sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
            pred = model(sent1, sent1_len, sent2, sent2_len)
        elif args.textcnn:
            sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
            y = y.float()
            pred = model(sent1, sent2)

        loss = model.loss_fn(pred, y)
        # backword
        loss.backward()
        optimizer.step()
        # scheluder.step()
        
        # get log
        loss_all.append(loss.item())
        loss_interval.append(loss.item())
        acc = model.get_acc(pred, y)
        acc_all.append(acc)
        acc_interval.append(acc)

        pbar.set_description('Epoch: %2d | Loss: %.3f | Accuracy: %.3f' \
            % (epoch, np.mean(loss_all), np.mean(acc_all)))
    
def evaluate_epoch(model, dataloader, epoch):
    logger.info('Epoch %2d: Evaluating...' % epoch)
    model.eval()
    acc = []
    loss_all = []
    with torch.no_grad():
        for batch in dataloader:
            if args.bert:
                idx, x, y, input_id = map(lambda x: x.to(device), batch)
                pred = model(x, input_id)
            elif args.esim:
                sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
                pred = model(sent1, sent1_len, sent2, sent2_len)
            elif args.bert_esim:
                sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
                pred = model(sent1, sent1_len, sent2, sent2_len)
            elif args.textcnn:
                sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
                y = y.float()
                pred = model(sent1, sent2)
            loss = model.loss_fn(pred, y) 
            loss_all.append(loss.item())
            acc.append(model.get_acc(pred, y))
    return np.mean(acc), np.mean(loss_all)

def train(model, optimizer, scheduler, train_loader, val_loader, num_epoch):
    for epoch in range(num_epoch):
        max_acc = 0

        t_1 = time.time()
        train_epoch(model, optimizer, scheduler, train_loader, epoch)
        acc, loss = evaluate_epoch(model, val_loader, epoch)
        t_2 = time.time()
        t = t_2 - t_1
        logger.info('FINISHED!    Epoch: %2d | Loss: %.2f | Accuracy: %.3f | Time Used: %2dmin%2ds | Speed: %4.2f items/s' \
            % (epoch, loss, acc, t//60, t%60, len(train_loader)/t))
        logger.info('')

        if acc > max_acc:
            save_model(model, optimizer, args.model_dir)
            max_acc = acc

def save_model(model, optimizer, model_dir):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state_dict, model_dir)

def main():
    args = parser.parse_args()

    # get dataset and dataloader
    train_data = load_pt(args.train_data)
    dev_data = load_pt(args.dev_data)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)

    # build model
    if args.bert:
        model = BaseBertModel(args.bert_type, 768, args.d_hidden, args.drop_out, 2)
    elif args.esim:
        model = ESIM(len(train_data.word2idx), args.embedding_dim, args.d_hidden, dropout=args.drop_out, num_classes=2, device='cuda')
    elif args.bert_esim:
        model = BertESIM(args.bert_type, args.embedding_dim, args.d_hidden, dropout=args.drop_out, num_classes=2, device='cuda')
    elif args.textcnn:
        # model = TextCNN(len(train_data.word2idx), args.embedding_dim, args.d_hidden, args.drop_out)
        model = TextCNN(0, args.embedding_dim, args.d_hidden, args.drop_out)
    else:
        assert(Exception('Please input the correct model type.'))
    model =  model.to(device)

    # build optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # warm_up = 500
    # cr = args.lr / log2(warm_up)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * log2(ee + 1) if ee < warm_up else args.lr)
    scheduler = None

    # begin train
    train(model, optimizer, scheduler, train_dataloader, dev_dataloader, args.num_epoch)

if __name__ == '__main__':
    main()
