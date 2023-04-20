from tqdm import tqdm
from model import bce_loss
from test import valid, test
import torch
from collections import deque
from transfer import collector
import numpy as np
import torch.nn as nn
import os
import time
import math

def train_one_batch_contrastive(batch, model, optimizer, gradient_norm_queue, prototype_queue, args):
    model.to(args.device)
    src, trg = batch
    input_id = src.to(args.device)
    trg = trg.to(args.device)
    optimizer.zero_grad()
    model.train()

    emb_out, lengths, masks = model.emb(input_id)
    representation = model.extractor(emb_out, lengths, masks).to(args.device)
    y_pred = model.clf(representation).to(args.device)
    loss = bce_loss(y_pred, trg.float())
    # print('y_pred.grad', y_pred.grad)
    loss2 = contrastive_loss(representation, trg.cpu(), prototype_queue, args)
    loss += args.contrastive_weight * loss2
    loss = loss.requires_grad_(True)
    loss.backward()
    clip_gradient(model, gradient_norm_queue, args)
    optimizer.step(closure=None)
    return loss.item()

def contrastive_loss(representation, traget, prototype_queue, args):
    # mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    representation = nn.functional.normalize(representation, dim=-1).to(args.device)
    queue = nn.functional.normalize(torch.from_numpy(prototype_queue), dim=-1).to(args.device).float().detach()
    # batch_queue = get_selected_proto(queue, traget, representation)
    batch_queue = queue
    loss = 0
    count = 0
    batch_size, label_size, hidden_size = representation.shape
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if traget[batch_idx, label_idx] ==1:
                q = representation[batch_idx,label_idx,:] #feat_size
                k = get_positive_proto(queue, label_idx) #feat_size, label_size*feat_size
                # temp_loss = mse_loss(q.view(1, q.size(-1)), k.view(1, k.size(-1)).float())

                # compute logits
                # positive logits: 1
                l_pos = torch.einsum('c,c->', [q, k])
                l_pos = l_pos.view(1,1)
                # negative logits: K
                l_neg = torch.einsum('kc,c->k', [batch_queue.clone().detach(), q])
                l_neg = l_neg.view(1, l_neg.size(0))
                # logits: (1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= args.T

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                count+=1 #count ~ 200
                temp_loss = ce_loss(logits, labels)
                loss+=temp_loss
    return loss/count/batch_size

def get_positive_proto(prototype_queue, label_idx):
    # return prototype_queue[label_idx], None
    return prototype_queue[label_idx]

def get_selected_proto(queue, target, representation):
    batch_size, label_size, hidden_size = representation.shape
    label_set = []
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if target[batch_idx, label_idx] ==1:
                if label_idx not in label_set:
                    label_set.append(label_idx)
    batch_index = torch.Tensor(label_set).cuda().long()
    batch_queue = torch.index_select(queue, 0, batch_index)
    return batch_queue

def clip_gradient(model, gradient_norm_queue, args):
    if args.gradient_clip_value is not None:
        max_norm = max(gradient_norm_queue)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * args.gradient_clip_value)
        gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
        if total_norm > max_norm * args.gradient_clip_value:
            print(F'Clipping gradients with total norm {round(float(total_norm.cpu().numpy()), 5)} '
                        F'and max norm {round(float(max_norm.cpu().numpy()), 5)}')