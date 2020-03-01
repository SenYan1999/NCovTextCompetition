import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class TextCNN(nn.Module):
    def __init__(self, vocab_size, d_embedding, d_hidden, drop_out):
        super(TextCNN, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embedding, padding_idx=0)
        self.embedding = BertModel.from_pretrained('pretrained_bert_model/')
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d_embedding, out_channels=d_embedding, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(p=drop_out)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=d_embedding, out_channels=d_embedding, kernel_size=4, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(p=drop_out)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=d_embedding, out_channels=d_embedding, kernel_size=6, stride=2),
            nn.MaxPool1d(kernel_size=9),
            nn.ReLU(),
            nn.Dropout(p=drop_out)
        )
        self.dense = nn.Linear(d_embedding, d_hidden)

    def forward(self, sent1, sent2):
        # sent1, sent2 = self.embedding(sent1)[0].permute(0, 2, 1), self.embedding(sent2)[0].permute(0, 2, 1)

        # sent1 = self.conv1(sent1)
        # sent1 = self.conv2(sent1)
        # sent1 = self.conv3(sent1)

        # sent2 = self.conv1(sent2)
        # sent2 = self.conv2(sent2)
        # sent2 = self.conv3(sent2)

        # sent1, sent2 = sent1.squeeze(), sent2.squeeze()
        sent1, sent2 = self.embedding(sent1)[1], self.embedding(sent2)[1]
        sent1, sent2 = self.dense(sent1), self.dense(sent2)
        cosine_dist = torch.cosine_similarity(sent1, sent2)

        return torch.sigmoid(cosine_dist)
    
    def loss_fn(self, pred, y):
        return F.binary_cross_entropy(pred, y)

    def get_acc(self, pred, y):
        pred = (pred > 0.5).long()
        y = y.long()
        return (torch.sum(pred == y).float() / pred.shape[0]).item()
        