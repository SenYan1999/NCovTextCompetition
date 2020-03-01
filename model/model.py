import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BaseBertModel(nn.Module):
    def __init__(self, bert_config, d_bert, d_hidden, p_drop, num_class):
        super(BaseBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('./pretrained_bert_model')
        self.bert_pool = nn.Linear(d_bert, d_bert)
        self.linear = nn.Linear(d_bert, d_hidden // 2)
        self.lstm = nn.LSTM(d_hidden, d_hidden // 2, dropout=p_drop, num_layers=2, batch_first=True, bidirectional=True)
        self.dense_layer = nn.Linear(d_hidden // 2, num_class)
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x, idx):
        x_mask = (x != 1).int()
        x_output = self.bert(x, attention_mask=x_mask, token_type_ids=idx)[1]
        # x_output = x_output[:, 0, :]
        x_output = self.drop(F.tanh(self.linear(x_output)))

        # original_length = x_mask.shape[1]
        # lengths = torch.sum(x_mask, dim=-1)
        # sort_length, sort_idx = torch.sort(lengths, descending=True)
        # x_pack = nn.utils.rnn.pack_padded_sequence(x_output[sort_idx], \
        #     sort_length, batch_first=True)
        # output, _ = self.lstm(x_pack)
        # output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=original_length)
        # _, unsorted_idx = torch.sort(sort_idx)
        # x_output = output[unsorted_idx]

        # x_output = torch.max_pool1d(x_output.permute(0, 2, 1), kernel_size=x_output.shape[1]).squeeze()
        x_output = self.dense_layer(x_output)
        return torch.log_softmax(x_output, dim=-1)
    
    def loss_fn(self, pred, y):
        return F.nll_loss(pred, y)

    def get_acc(self, pred, ground_truth):
        pred = torch.argmax(pred, dim=-1)
        acc = (pred == ground_truth)
        return torch.sum(acc).item() / ground_truth.shape[0]
