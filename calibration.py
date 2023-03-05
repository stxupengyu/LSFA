from tqdm import tqdm
import torch.nn as nn
from model import bce_loss
from test import evaluate_valid, evaluate_test
import torch
from collections import deque
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import time
import math

def calibrate(model, new_model, new_optimizer, train_loader, val_loader, test_loader, calibration_loader, mlb, args):
    model.to(args.device)
    fix_model(model)
    model.eval()
    new_model.to(args.device)
    new_model.train()

    global_step = 0
    num_stop_dropping = 0
    best_valid_result = 0
    step = 100
    gradient_norm_queue = deque([np.inf], maxlen=5)
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader, 1):
            global_step += 1
            batch_loss = train_one_batch(batch, model, new_model, new_optimizer, gradient_norm_queue, args)
            if global_step % step == 0:
                valid_result = valid(model, new_model, val_loader, mlb, args)[-1]
                if valid_result > best_valid_result:
                    best_valid_result = valid_result
                    num_stop_dropping = 0
                    torch.save(new_model.state_dict(), args.check_pt_new_model_path)
                else:
                    num_stop_dropping += 1
                if args.test_each_epoch:
                    valid_result = valid(model, new_model, val_loader, mlb, args)
                    test_result = test(model, new_model, test_loader, mlb, args)
                    print(f'C | Epoch: {epoch} | Loss: {batch_loss: .4f} | Stop: {num_stop_dropping} | Valid: {valid_result} | Test: {test_result} ')
                else:
                    valid_result = valid(model, new_model, val_loader, mlb, args)
                    print(f'C | Epochs: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | Valid Result: {valid_result}')

        if epoch >= args.calibration_warmup:#calibration warm up
            for i, batch in enumerate(calibration_loader, 1):
                calibrate_one_batch(batch, new_model, new_optimizer, gradient_norm_queue, args)
            if args.test_each_epoch:
                valid_result = valid(model, new_model, val_loader, mlb, args)
                test_result = test(model, new_model, test_loader, mlb, args)
                print(f'C | Epoch: {epoch} | Loss: {batch_loss: .4f} | Stop: {num_stop_dropping} | Valid: {valid_result} | Test: {test_result} ')
            else:
                valid_result = valid(model, new_model, val_loader, mlb, args)
                print(f'C | Epochs: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | Valid Result: {valid_result}')


        if num_stop_dropping >= args.calibration_early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break


def calibrate_one_batch(batch, new_model, new_optimizer, gradient_norm_queue, args):
    new_model.to(args.device)
    representation, label_idx = batch
    representation = torch.unsqueeze(representation, 1).to(args.device) # batch_size*1*hidden_size
    representation = representation.repeat(1, args.label_size, 1).detach() # batch_size*label_size*hidden_size
    # assert torch.sum(torch.add(representation[:,0,:],-representation[:,-1,:]))==0 #make sure the repeat operation
    label_idx = label_idx.to(args.device)
    new_optimizer.zero_grad()
    new_model.train()
    # print(representation, 'representation') #make sure representaton without gradient
    # print(representation.shape, 'representation.shape')
    y_pred = new_model(representation)
    loss = args.calibration_weight*calibration_loss(y_pred, label_idx)
    loss.backward()
    #clip_gradient(model, gradient_norm_queue, args)
    new_optimizer.step(closure=None)

def calibration_loss(y_pred, label_idx):
    trg = torch.sigmoid(y_pred).detach() #without gradient
    for i, idx in enumerate(label_idx):
        trg[i,int(idx)] = 1
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(y_pred, trg.float())
    return loss

def train_one_batch(batch, model, new_model, new_optimizer, gradient_norm_queue, args):
    # train for one batch
    model.to(args.device)
    src, trg = batch
    input_id = src.to(args.device)
    trg = trg.to(args.device)
    new_optimizer.zero_grad()
    model.train()

    emb_out, lengths, masks = model.emb(input_id)
    representation = model.extractor(emb_out, lengths, masks).to(args.device)
    y_pred = new_model(representation)

    loss = bce_loss(y_pred, trg.float())
    loss.backward()
    clip_gradient(model, gradient_norm_queue, args)
    new_optimizer.step(closure=None)
    return loss.item()

def clip_gradient(model, gradient_norm_queue, args):
    if args.gradient_clip_value is not None:
        max_norm = max(gradient_norm_queue)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * args.gradient_clip_value)
        gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
        if total_norm > max_norm * args.gradient_clip_value:
            print(F'Clipping gradients with total norm {round(float(total_norm.cpu().numpy()), 5)} '
                        F'and max norm {round(float(max_norm.cpu().numpy()), 5)}')

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def valid(model, new_model, test_data_loader, mlb, args):
    model.to(args.device)
    model.eval()
    new_model.to(args.device)
    new_model.eval()
    pre_K = 10
    y_test = None
    y_pred = None
    with torch.no_grad():
        for batch_i, batch in enumerate(test_data_loader):
            src, trg = batch
            # move data to GPU if available
            input_id = src.to(args.device)
            test_label = trg.to(args.device)

            emb_out, lengths, masks = model.emb(input_id)
            representation = model.extractor(emb_out, lengths, masks).to(args.device)
            output = new_model(representation)

            if y_test is None:
                y_test = test_label
                y_pred = output
            else:
                y_test = torch.cat((y_test, test_label), 0)
                y_pred = torch.cat((y_pred, output), 0)
    y_scores, y_pred = torch.topk(y_pred, pre_K)
    y_test = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    result = evaluate_valid(y_test, y_pred, mlb, args)
    return result

def test(model, new_model, test_loader, mlb, args):
    model.to(args.device)
    model.eval()
    new_model.to(args.device)
    new_model.eval()
    pre_K = 10
    y_pred = None
    with torch.no_grad():
        for i, [src] in enumerate(test_loader):
            input_id = src.to(args.device)

            emb_out, lengths, masks = model.emb(input_id)
            representation = model.extractor(emb_out, lengths, masks).to(args.device)
            output = new_model(representation)

            if y_pred is None:
                y_pred = output
            else:
                y_pred = torch.cat((y_pred, output), 0)
    scores, labels = torch.topk(y_pred, pre_K)
    scores = torch.sigmoid(scores).cpu().numpy()
    labels = labels.cpu().numpy()

    labels = mlb.classes_[labels]
    test_labels = np.load(os.path.join(args.data_dir, args.test_labels), allow_pickle=True)

    mlb = MultiLabelBinarizer(sparse_output=True)
    y_test = mlb.fit_transform(test_labels)

    result = evaluate_test(y_test, labels, mlb, args)
    return result