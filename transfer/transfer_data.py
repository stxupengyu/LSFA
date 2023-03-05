import random
import torch.utils.data as data_utils
import torch
import numpy as np
from tqdm import tqdm

def get_dataset(feature_dict, head_list, prototype_dict, args):
    x_feats = None
    y_pro = None
    for label_idx in tqdm(feature_dict.keys()):
        if label_idx in head_list:
            for instance in feature_dict[label_idx]:
                pro = prototype_dict[label_idx]
                if x_feats is None:
                    x_feats = instance
                    y_pro = pro
                else:
                    x_feats = np.vstack((x_feats, instance))
                    y_pro = np.vstack((y_pro, pro))
    print('Dataset size of VAE:', x_feats.shape[0])

    #shuffle
    x_train, x_valid, y_train, y_valid = get_shuffle(x_feats, y_pro, args)
    print(F'x_train.shape: {x_train.shape}')
    print(F'y_train.shape: {y_train.shape}')
    print(F'x_train.shape: {x_valid.shape}')
    print(F'y_valid.shape: {y_valid.shape}')

    train_data = data_utils.TensorDataset(torch.from_numpy(x_train).type(torch.LongTensor), torch.from_numpy(y_train).type(torch.Tensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(x_valid).type(torch.LongTensor), torch.from_numpy(y_valid).type(torch.Tensor))

    train_vae_loader = data_utils.DataLoader(train_data, args.vae_batch_size, shuffle=True, drop_last=True)
    valid_vae_loader = data_utils.DataLoader(val_data, args.vae_batch_size, shuffle=True, drop_last=True)

    return train_vae_loader, valid_vae_loader

def get_shuffle(x, y, args):
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    split_num = int(0.9*x.shape[0])
    x_train, x_valid = x[:split_num], x[split_num:]
    y_train, y_valid = y[:split_num], y[split_num:]
    return x_train, x_valid, y_train, y_valid