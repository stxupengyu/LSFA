import re
import os
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim.models import KeyedVectors
from tqdm import tqdm
from typing import Union, Iterable
import logging
import argparse
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# preprocessing
parser.add_argument("--data_dir", default="/data/pengyu/eurlex", type=str,
                    help="The input data directory")
parser.add_argument("--raw_train_text", default="train_texts.txt", type=str,
                    help="data before preprocessing")
parser.add_argument("--raw_train_label", default="train_labels.txt", type=str,
                    help="data before preprocessing")
parser.add_argument("--raw_test_text", default="test_texts.txt", type=str,
                    help="data before preprocessing")
parser.add_argument("--raw_test_label", default="test_labels.txt", type=str,
                    help="data before preprocessing")
parser.add_argument("--vocab_path", default="vocab.npy", type=str,
                    help="path of vocabulary")
parser.add_argument("--w2v_model", default="/data/pengyu/glove.840B.300d.gensim", type=str,
                    help="path of Gensim Word2Vec model")
parser.add_argument("--emb_init", default="emb_init.npy", type=str,
                    help="embedding layer from glove")
parser.add_argument('--max_len', type=int, default=500,
                    help="max length of document")
parser.add_argument('--vocab_size', type=int, default=500000,
                    help="vocabulary size of dataset")
args = parser.parse_args()

def preprocessing(args):
    raw_train_text = os.path.join(args.data_dir, args.raw_train_text)
    raw_train_label = os.path.join(args.data_dir, args.raw_train_label)
    raw_test_text = os.path.join(args.data_dir, args.raw_test_text)
    raw_test_label = os.path.join(args.data_dir, args.raw_test_label)
    vocab_path = os.path.join(args.data_dir, args.vocab_path)
    emb_path = os.path.join(args.data_dir, args.emb_init)
    max_len = args.max_len
    vocab_size = args.vocab_size
    w2v_model = args.w2v_model

    logger.info(F'Building Vocab. {raw_train_text}')
    with open(raw_train_text) as fp:
        vocab, emb_init = build_vocab(fp, w2v_model, vocab_size=vocab_size)
    np.save(vocab_path, vocab)
    np.save(emb_path, emb_init)
    vocab = {word: i for i, word in enumerate(vocab)}
    logger.info(F'Vocab Size: {len(vocab)}')

    logger.info(F'Getting Training Dataset: {raw_train_text} Max Length: {max_len}')
    texts, labels = convert_to_binary(raw_train_text,raw_train_label,max_len, vocab)
    logger.info(F'Size of Samples: {len(texts)}')
    np.save(os.path.splitext(raw_train_text)[0], texts)
    np.save(os.path.splitext(raw_train_label)[0], labels)

    logger.info(F'Getting Test Dataset: {raw_test_text} Max Length: {max_len}')
    texts, labels = convert_to_binary(raw_test_text,raw_test_label,max_len, vocab)

    logger.info(F'Size of Samples: {len(texts)}')
    np.save(os.path.splitext(raw_test_text)[0], texts)
    np.save(os.path.splitext(raw_test_label)[0], labels)

def tokenize(sentence: str, sep='/SEP/'):
    return [token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]

def build_vocab(texts: Iterable, w2v_model: Union[KeyedVectors, str], vocab_size=500000,
                pad='<PAD>', unknown='<UNK>', sep='/SEP/', max_times=1, freq_times=1):
    if isinstance(w2v_model, str):
        w2v_model = KeyedVectors.load(w2v_model)
    emb_size = w2v_model.vector_size
    vocab, emb_init = [pad, unknown], [np.zeros(emb_size), np.random.uniform(-1.0, 1.0, emb_size)]
    counter = Counter(token for t in texts for token in set(t.split()))
    for word, freq in sorted(counter.items(), key=lambda x: (x[1], x[0] in w2v_model), reverse=True):
        if word in w2v_model or freq >= freq_times:
            vocab.append(word)
            # We used embedding of '.' as embedding of '/SEP/' symbol.
            word = '.' if word == sep else word
            emb_init.append(w2v_model[word] if word in w2v_model else np.random.uniform(-1.0, 1.0, emb_size))
        if freq < max_times or vocab_size == len(vocab):
            break
    return np.asarray(vocab), np.asarray(emb_init)

def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    if max_len is None:
        return texts
    texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts

def convert_to_binary(text_file, label_file=None, max_len=None, vocab=None, pad='<PAD>', unknown='<UNK>'):
    with open(text_file) as fp:
        texts = np.asarray([[vocab.get(word, vocab[unknown]) for word in line.split()]
                           for line in tqdm(fp, desc='Converting token to id', leave=False)])
    labels = None
    if label_file is not None:
        with open(label_file) as fp:
            labels = np.asarray([[label for label in line.split()]
                                 for line in tqdm(fp, desc='Converting labels', leave=False)])
    return truncate_text(texts, max_len, vocab[pad], vocab[unknown]), labels

if __name__ == '__main__':
    preprocessing(args)