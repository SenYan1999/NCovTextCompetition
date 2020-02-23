import torch
import torch.nn as nn
from transformers import BertModel

class BaseBertModel(nn.Module):
    def __init__(self, bert_config, d_bert, d_hidden, p_drop, num_class):
        super(BaseBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('pretrained_bert_model/')
        self.linear = nn.Linear(d_bert, d_hidden)
        self.lstm = nn.LSTM(d_hidden, d_hidden // 2, dropout=p_drop, num_layers=2, batch_first=True, bidirectional=True)
        self.dense_layer = nn.Linear(d_hidden, num_class)

    def forward(self, x, idx):
        x_mask = (x != 1).int()
        x_output = self.bert(x, attention_mask=x_mask, token_type_ids=idx)[0]
        x_output = self.linear(x_output)

        original_length = x_mask.shape[1]
        lengths = torch.sum(x_mask, dim=-1)
        sort_length, sort_idx = torch.sort(lengths, descending=True)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_output[sort_idx], \
            sort_length, batch_first=True)
        output, _ = self.lstm(x_pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=original_length)
        _, unsorted_idx = torch.sort(sort_idx)
        x_output = output[unsorted_idx]

        x_output = x_output[:, 0, :]
        x_output = self.dense_layer(x_output)
        return torch.log_softmax(x_output, dim=-1)
