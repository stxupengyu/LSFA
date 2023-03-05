from tqdm import tqdm
import torch
import numpy as np
import os
import time
import math

def get_prototype(feature_dict):
    prototype_dict = {}
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:
            prototype = np.mean(feature_dict[label_idx], 0)
            prototype_dict[label_idx] = prototype
        else:
            prototype_dict[label_idx] = None
    return prototype_dict

def get_head(feature_dict, args):
    head_list = []
    tail_list = []
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:
            instance_num = len(feats)
            if instance_num > args.threshold:
                head_list.append(label_idx)
            else:
                tail_list.append(label_idx)
    return head_list, tail_list

def collect(model, train_loader, args):
    model.to(args.device)
    fix_model(model)
    model.eval()

    feature_dict = {}
    for idx in range(args.label_size):
        feature_dict[idx] = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 1):
            src, trg = batch
            # print(trg.shape, 'trg.shape')
            input_id = src.to(args.device)
            trg = trg.to(args.device)
            emb_out, lengths, masks = model.emb(input_id)
            representation = model.extractor(emb_out, lengths, masks).to(args.device)
            feature_dict = dict_append_batch(representation, trg, feature_dict)
            # break
    np.save(args.feature_dict_path, feature_dict)
    return feature_dict

def dict_append_batch(representation, trg, feature_dict):
    batch_size, label_size, hidden_size = representation.shape
    # print(representation.shape, 'representation.shape')
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if trg[batch_idx, label_idx] ==1:
                vector = torch.unsqueeze(representation[batch_idx,label_idx,:], 0) #1*hidden_size
                vector = vector.detach().cpu().numpy()
                feature_dict = dict_append_each(vector, label_idx, feature_dict)
    return feature_dict

def dict_append_each(vector, label_idx, feature_dict):
    if feature_dict[label_idx] is None:
        feature_dict[label_idx] = vector
    else:
        feature_dict[label_idx] = np.vstack((feature_dict[label_idx], vector)) #torch.cat((feature_dict[label_idx], vector), 0)
    return feature_dict

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_queue(feature_dict, args):
    queue = []
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:
            prototype = np.mean(feature_dict[label_idx], 0)#1*feat_size
        else:
            prototype = np.zeros(args.feat_size) #zero for the lack label
        queue.append(prototype)
    queue = np.array(queue)
    return queue
