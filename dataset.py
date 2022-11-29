import os
import numpy as np
import torch.utils.data as data_utils
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import logging
from scipy.sparse import csr_matrix

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(args, sample_level_da=None):
    train_texts = np.load(os.path.join(args.data_dir, args.train_texts), allow_pickle=True)
    train_labels = np.load(os.path.join(args.data_dir, args.train_labels), allow_pickle=True)
    test_texts = np.load(os.path.join(args.data_dir, args.test_texts), allow_pickle=True)
    emb_init = get_word_emb(os.path.join(args.data_dir, args.emb_init))
    X_train, X_valid, train_y, valid_y = train_test_split(train_texts, train_labels,
                                                                    test_size=args.valid_size,
                                                                    random_state=args.seed)

    mlb = get_mlb(os.path.join(args.data_dir, args.labels_binarizer), np.hstack((train_y, valid_y)))

    y_train, y_valid = mlb.transform(train_y), mlb.transform(valid_y)
    args.label_size = len(mlb.classes_)

    if args.sample_level_da==True:
        X_train, y_train = slda(X_train, y_train)

    logger.info(F'Size of Training Set: {len(X_train)}')
    logger.info(F'Size of Validation Set: {len(X_valid)}')

    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train.A).type(torch.LongTensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(X_valid).type(torch.LongTensor),
                                          torch.from_numpy(y_valid.A).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(test_texts).type(torch.LongTensor))

    train_loader = data_utils.DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = data_utils.DataLoader(val_data, args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = data_utils.DataLoader(test_data, args.batch_size, drop_last=False)

    return train_loader, val_loader, test_loader, emb_init, mlb, args

def get_word_emb(vec_path, vocab_path=None):
    if vocab_path is not None:
        with open(vocab_path) as fp:
            vocab = {word: idx for idx, word in enumerate(fp)}
        return np.load(vec_path, allow_pickle=True), vocab
    else:
        return np.load(vec_path, allow_pickle=True)

def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb

def slda(X_train, y_train):
    #sample level data augemtation
    y_train = y_train.A
    x = []
    y = []
    for i in range(10):
        x.append(X_train)
        y.append(y_train)
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (-1, x.shape[-1]))
    y = np.reshape(y, (-1, y.shape[-1]))
    y = csr_matrix(y)
    return x, y