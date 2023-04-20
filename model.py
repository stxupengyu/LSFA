import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Encoder, Head

class LSFL(nn.Module):

    def __init__(self, emb_init, args):
        super(LSFL, self).__init__()
        self.emb = Embedding(emb_size=args.emb_size, emb_init=emb_init, emb_trainable=args.emb_trainable)
        self.extractor = Extractor(args)
        self.clf = Classifier(args)

    def forward(self, input_id):
        emb_out, lengths, masks = self.emb(input_id)
        representation = self.extractor(emb_out, lengths, masks)
        logit = self.clf(representation)
        return logit

class Embedding(nn.Module):
    def __init__(self, vocab_size=None, emb_size=None, emb_init=None, emb_trainable=True, padding_idx=0, dropout=0.2):
        super(Embedding, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, inputs):
        emb_out = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        return emb_out[:, :lengths.max()], lengths, masks[:, :lengths.max()]

class Extractor(nn.Module):
    def __init__(self, args):
        super(Extractor,self).__init__()
        self.encoder = Encoder(args)
        self.head = Head(args)

    def forward(self, inputs, lengths, masks):
        representation = self.encoder(inputs, lengths, masks)
        representation = self.head(representation)
        return representation

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.classifier_mode = args.classifier_mode
        if self.classifier_mode=="type1":
            self.output_layer = torch.nn.Linear(args.feat_size, 1, bias=False)
            nn.init.xavier_uniform_(self.output_layer.weight)
        else:
            self.output_layer = torch.nn.Parameter(torch.randn(args.label_size, args.feat_size))
            nn.init.xavier_uniform_(self.output_layer)

    def forward(self, representation): #N*L*D
        if self.classifier_mode=="type1":
            return torch.squeeze(self.output_layer(representation),-1)
        else:
            return torch.sum(representation * self.output_layer, -1)

def bce_loss(y_pred, y_true):
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(y_pred, y_true)
    return loss

