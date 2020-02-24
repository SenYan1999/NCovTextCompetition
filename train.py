import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BaseBertModel, ESIM
from args import parser
from utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_accuracy(pred, ground_truth):
    pred = torch.argmax(pred, dim=-1)
    acc = (pred == ground_truth)
    return torch.sum(acc).item() / ground_truth.shape[0]

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
            x, y, idx = map(lambda x: x.to(device), batch)
            pred = model(x, idx)
        elif args.esim:
            sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
            pred = model(sent1, sent1_len, sent2, sent2_len)

        loss = F.nll_loss(pred, y)
        # backword
        loss.backward()
        optimizer.step()
        # scheluder.step()
        
        # get log
        loss_all.append(loss.item())
        loss_interval.append(loss.item())
        acc = get_accuracy(pred, y)
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
                x, y, idx = map(lambda x: x.to(device), batch)
                pred = model(x, idx)
            elif args.esim:
                sent1, sent1_len, sent2, sent2_len, y = map(lambda x: x.to(device), batch)
                pred = model(sent1, sent1_len, sent2, sent2_len)
            loss = F.nll_loss(pred, y) 
            loss_all.append(loss.item())
            acc.append(get_accuracy(pred, y))
    return np.mean(acc), np.mean(loss_all)

def train(model, optimizer, scheduler, train_loader, val_loader, num_epoch):
    for epoch in range(num_epoch):
        t_1 = time.time()
        train_epoch(model, optimizer, scheduler, train_loader, epoch)
        acc, loss = evaluate_epoch(model, val_loader, epoch)
        t_2 = time.time()
        t = t_2 - t_1
        logger.info('FINISHED!    Epoch: %2d | Loss: %.2f | Accuracy: %.2f | Time Used: %2dmin%2ds | Speed: %4.2f items/s' \
            % (epoch, loss, acc, t//60, t%60, len(train_loader)/t))
        logger.info('')

        torch.save(model, args.model_dir)


def main():
    args = parser.parse_args()

    # get dataset and dataloader
    train_data = load_pt(args.train_data)
    dev_data = load_pt(args.dev_data)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)

    # build model
    if args.bert:
        model = BaseBertModel(args.bert_type, 768, args.d_hidden, args.drop_out, 2)
    elif args.esim:
        model = ESIM(len(train_data.word2idx), args.embedding_dim, args.d_hidden, dropout=0.5, num_classes=2, device='cuda')
    else:
        assert(Exception('Please input the correct model type.'))
    model =  model.to(device)

    # build optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    # begin train
    train(model, optimizer, scheduler, train_dataloader, dev_dataloader, args.num_epoch)

if __name__ == '__main__':
    main()
