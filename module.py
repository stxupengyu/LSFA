import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder,self).__init__()
        self.lstm = torch.nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size, batch_first=True, bidirectional=True)
        self.init_state = nn.Parameter(torch.zeros(2 * 2, 1, args.hidden_size))
        self.dropout = nn.Dropout(args.dropout)

        self.attention = nn.Linear(args.hidden_size*2, args.label_size, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, lengths, masks):
        self.lstm.flatten_parameters()
        init_state = self.init_state.repeat([1, inputs.size(0), 1])
        cell_init, hidden_init = init_state[:init_state.size(0) // 2], init_state[init_state.size(0) // 2:]
        idx = torch.argsort(lengths, descending=True)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs[idx], lengths[idx].cpu(), batch_first=True)
        temp = self.lstm(packed_inputs, (hidden_init, cell_init))[0]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)
        outputs = self.dropout(outputs[torch.argsort(idx)])

        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        attention = self.attention(outputs).transpose(1, 2).masked_fill(~masks, -np.inf)  # N, labels_num, L
        attention = F.softmax(attention, -1)
        # attention = Sparsemax(dim=-1)(attention)
        representation = attention @ outputs   # N, labels_num, hidden_size
        return representation

class Head(nn.Module):

    def __init__(self, args):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.feat_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, representation):
        feats = self.head(representation)
        return feats

# class Head(nn.Module):
#
#     def __init__(self, args):
#         super(Head, self).__init__()
#         self.head = nn.Sequential(
#             nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(args.hidden_size * 2, args.feat_size)
#         )
#
#     def forward(self, representation):
#         feats = F.normalize(self.head(representation), dim=-1)
#         return feats

