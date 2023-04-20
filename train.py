from tqdm import tqdm
from model import bce_loss
from test import valid, test
import torch
from collections import deque
from transfer import collector
from contrastive import train_one_batch_contrastive
import numpy as np
import torch.nn as nn
import os
import time
import math

def train(model, optimizer,train_loader, val_loader, test_loader, mlb, args):

    state = {}
    global_step = 0
    num_stop_dropping = 0
    best_valid_result = 0
    step = 100
    gradient_norm_queue = deque([np.inf], maxlen=5)
    for epoch in range(args.epochs):
        if args.swa_mode==True and epoch==args.swa_warmup:
            swa_init(state, model)
        if args.contrastive_mode==True and epoch>=args.contrastive_warmup and epoch%2==0:
            if args.pre_feature_dict==True:
                feature_dict = np.load(args.pre_feature_dict_path, allow_pickle=True).item()  # sub optimal
            else:
                collector.collect(model, train_loader, args)
                feature_dict = np.load(args.feature_dict_path, allow_pickle=True).item()
            prototype_queue = collector.get_queue(feature_dict, args)
        for i, batch in enumerate(train_loader, 1):
            global_step += 1
            if args.contrastive_mode==True and epoch>=args.contrastive_warmup and epoch%2==0:
                batch_loss = train_one_batch_contrastive(batch, model, optimizer, gradient_norm_queue, prototype_queue, args)
            else:
                batch_loss = train_one_batch(batch, model, optimizer, gradient_norm_queue, args)
            if global_step % step == 0:
                if args.swa_mode == True:
                    swa_step(state, model)
                    swap_swa_params(state, model)
                valid_result = valid(model, val_loader, mlb, args)[-1]
                if valid_result > best_valid_result:
                    best_valid_result = valid_result
                    num_stop_dropping = 0
                    torch.save(model.state_dict(),args.check_pt_model_path)
                else:
                    num_stop_dropping += 1

                if args.swa_mode == True:
                    swap_swa_params(state, model)

                if args.test_each_epoch:
                    valid_result = valid(model, val_loader, mlb, args)
                    test_result = test(model, test_loader, mlb, args)
                    print(f'Epoch: {epoch} | Loss: {batch_loss: .4f} | Stop: {num_stop_dropping} | Valid: {valid_result} | Test: {test_result} ')
                else:
                    valid_result = valid(model, val_loader, mlb, args)
                    print(f'Epochs: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | Valid Result: {valid_result}')

        if num_stop_dropping >= args.early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break

def train_one_batch(batch, model, optimizer, gradient_norm_queue, args):
    # train for one batch
    model.to(args.device)
    src, trg = batch
    input_id = src.to(args.device)
    trg = trg.to(args.device)
    optimizer.zero_grad()
    model.train()
    y_pred = model(input_id)
    loss = bce_loss(y_pred, trg.float()).requires_grad_(True)
    loss.backward()
    clip_gradient(model, gradient_norm_queue, args)
    optimizer.step(closure=None)
    return loss.item()

def clip_gradient(model, gradient_norm_queue, args):
    if args.gradient_clip_value is not None:
        max_norm = max(gradient_norm_queue)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * args.gradient_clip_value)
        gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
        if total_norm > max_norm * args.gradient_clip_value:
            print(F'Clipping gradients with total norm {round(float(total_norm.cpu().numpy()), 5)} '
                        F'and max norm {round(float(max_norm.cpu().numpy()), 5)}')

def swa_init(state, model):
    print('SWA Initializing')
    swa_state = state['swa'] = {'models_num': 1}
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()

def swa_step(state, model):
    if 'swa' in state:
        swa_state = state['swa']
        swa_state['models_num'] += 1
        beta = 1.0 / swa_state['models_num']
        with torch.no_grad():
            for n, p in model.named_parameters():
                swa_state[n].mul_(1.0 - beta).add_(beta, p.data)

def swap_swa_params(state, model):
    if 'swa' in state:
        swa_state = state['swa']
        for n, p in model.named_parameters():
            p.data, swa_state[n] = swa_state[n], p.data



