import os
import logging
import argparse
import time
import random
import numpy as np
import dataset, train, test
from optimizers import *
from model import Classifier, LSFL
from transfer import collector, generator, transfer_model, transfer_train, transfer_data
import calibration
from utils import time_since, cprint
import torch
from torch.optim import Adam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument("--data_dir", default="/data/eurlex", type=str,
                        help="The input data directory")
    parser.add_argument("--train_texts", default="train_texts.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--train_labels", default="train_labels.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--test_texts", default="test_texts.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--test_labels", default="test_labels.npy", type=str,
                        help="data after preprocessing")
    parser.add_argument("--vocab_path", default="vocab.npy", type=str,
                        help="data before preprocessing")
    parser.add_argument("--emb_init", default="emb_init.npy", type=str,
                        help="embedding layer from glove")
    parser.add_argument("--labels_binarizer", default="labels_binarizer", type=str,
                        help="")
    parser.add_argument('--max_len', type=int, default=500,
                        help="max length of document")
    parser.add_argument('--vocab_size', type=int, default=500000,
                        help="vocabulary size of dataset")
    parser.add_argument('--valid_size', type=int, default=200,
                        help="size of validation set")

    #model
    parser.add_argument('--emb_trainable', type=bool, default=False,
                        help="train the embedding layer")
    parser.add_argument('--emb_size', type=int, default=300,
                        help="embedding size")
    parser.add_argument('--hidden_size', type=int, default=256,
                        help="hideden size of LSTM")
    parser.add_argument('--feat_size', type=int, default=300,
                        help="feature size of LSFL")
    parser.add_argument('--classifier_mode', type=str, default='type1', #'type1' or 'type2'
                        help="type of classifiers")
    parser.add_argument("--dropout", default=0.5, required=False, type=float,
                        help="dropout of LSFL")
    parser.add_argument("--learning_rate", default=1e-3, required=False, type=float,
                        help="learning rate of LSFL")

    #training
    parser.add_argument('--gpuid', type=int, default=7,
                        help="gpu id")
    parser.add_argument('--epochs', type=int, default=100,
                        help="epoch of LSFL")
    parser.add_argument('--early_stop_tolerance', type=int, default=15,
                        help="early stop of LSFL")
    parser.add_argument('--batch_size', type=int, default=40,
                        help="batch size of LSFL")
    parser.add_argument('--swa_warmup', type=int, default=10,
                        help="begin epoch of swa")
    parser.add_argument('--swa_mode', type=bool, default=True,
                        help="use swa strategy")
    parser.add_argument('--gradient_clip_value', type=int, default=5.0,
                        help="gradient clip")
    parser.add_argument('--seed', type=int, default=100,
                        help="random seed for initialization")
    parser.add_argument('--test_each_epoch', type=bool, default=False,#True False
                        help="test performance on each epoch")
    parser.add_argument('--report_psp', type=bool, default=True,
                        help="report psp metric")

    #VAE&Augmentaion
    parser.add_argument('--threshold', type=int, default=50,
                        help="head to tail threshold")
    parser.add_argument('--da_number', type=int, default=10,
                        help="times of augmentation")
    parser.add_argument('--vae_epochs', type=int, default=100,
                        help="eopch of VAE")
    parser.add_argument('--vae_batch_size', type=int, default=40,
                        help="batch size of VAE")
    parser.add_argument('--vae_early_stop_tolerance', type=int, default=10,
                        help="early stop of VAE")
    parser.add_argument("--vae_learning_rate", default=1e-4, required=False, type=float,
                        help="learning rate of VAE")

    #adjustment(calibartion)
    parser.add_argument('--calibration_warmup', type=int, default=5,
                        help="begin epoch of adjustment")
    parser.add_argument('--calibration_batch_size', type=int, default=40,
                        help="batch size of adjustment")
    parser.add_argument('--calibration_early_stop_tolerance', type=int, default=10,
                        help="early stop of adjustment")
    parser.add_argument("--calibration_learning_rate", default=1e-3, required=False, type=float,
                        help="learning rate of adjustment")
    parser.add_argument('--calibration_weight', type=float, default=1,
                        help="weight of adjustment")

    #contrastive
    parser.add_argument('--contrastive_mode', type=bool, default=True , #True, False
                        help="use contrastive learning")
    parser.add_argument('--contrastive_warmup', type=int, default=25, #25
                        help="begin epoch of contrastive learning")
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help="weight of contrastive learning")
    parser.add_argument('--contrastive_batch_size', type=int, default=40,
                        help="batch size of contrastive learning")
    parser.add_argument('--T', type=float, default=0.07,
                        help="temperature of contrastive learning")

    #mode&flexible
    parser.add_argument('--pretrained', type=bool, default=False,#True False
                        help="use pretrained LSFL model")
    parser.add_argument("--pretrained_path", default='/data/eurlex/model/lstm_20220818-141629.pth', type=str,
                        help="path of pretrained LSFL model")
    parser.add_argument('--pre_feature_dict', type=bool, default=False,#True False
                        help="use former feature dictionary")
    parser.add_argument("--pre_feature_dict_path", default='/data/eurlex/model/feature_dict_20220818-135926.npy', type=str,
                        help="path of former feature dictionary")
    parser.add_argument('--pre_vaed', type=bool, default=False,
                        help="use pretrained VAE model")
    parser.add_argument("--pre_vaed_path", default='/data/eurlex/model/vae_20220818-141629.pth', type=str,
                        help="path of pretrained VAE model")
    parser.add_argument('--pre_calibrated', type=bool, default=False,
                        help="use pre-adjusted classifiers")
    parser.add_argument("--pre_calibrated_path", default='/data/EUR-Lex/model/clf_20220809-131955.pth', type=str,
                        help="path of adjusted classifiers")
    parser.add_argument('--save_da_data', type=bool, default=False,
                        help="save da dictionary for plot")
    parser.add_argument('--save_prediction', type=bool, default=False,
                        help="save prediction for plot")
    parser.add_argument('--lsan', type=bool, default=False,
                        help="use lsan")
    parser.add_argument('--sample_level_da', type=bool, default=False,
                        help="try sample level da")

    args = parser.parse_args()

    args.model_path = os.path.join(args.data_dir, 'model')
    os.makedirs(args.model_path, exist_ok=True)
    args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    args.check_pt_model_path = os.path.join(args.model_path, "lstm_%s.pth" % args.timemark)
    args.check_pt_new_model_path = os.path.join(args.model_path, "clf_%s.pth" % args.timemark)
    args.check_pt_vae_model_path = os.path.join(args.model_path, "vae_%s.pth" % args.timemark)
    args.feature_dict_path = os.path.join(args.model_path, "feature_dict_%s.npy" % args.timemark)
    args.da_dict_path = os.path.join(args.model_path, "da_dict_%s.npy" % args.timemark)
    args.prediction_path = os.path.join(args.model_path, "prediction_%s.npy" % args.timemark)
    cprint('args', args)

    #for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    da4mltc(args)

def da4mltc(args):

    #Dataset
    start_time = time.time()
    logger.info('Data Loading')
    train_loader, val_loader, test_loader, emb_init, mlb, args = dataset.get_data(args)
    load_data_time = time_since(start_time)
    logger.info('Time for loading the data: %.1f s' %load_data_time)

    #Model
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    args.device = torch.device('cuda:0')
    model = LSFL(emb_init, args)
    model = model.to(args.device)
    # optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    optimizer = DenseSparseAdam(model.parameters())

    #Training
    if args.pretrained==False:
        train.train(model, optimizer, train_loader, val_loader, test_loader, mlb, args)
        training_time = time_since(start_time)
        logger.info('Time for training: %.1f s' % training_time)
        logger.info(f'Best Model Path: {args.check_pt_model_path}')
    else:
        args.check_pt_model_path = args.pretrained_path
    model.load_state_dict(torch.load(args.check_pt_model_path, map_location=args.device))

    #Collecting
    if args.pre_feature_dict == False:
        logger.info('Collecting')
        start_time = time.time()
        feature_dict = collector.collect(model, train_loader, args)
        logger.info(f'Collected Feature Dictionary Path: {args.feature_dict_path}')
        logger.info('Time for Collecting: %.1f s' % time_since(start_time))
    else:
        feature_dict = np.load(args.pre_feature_dict_path, allow_pickle=True).item()
    prototype_dict = collector.get_prototype(feature_dict)
    head_list, tail_list = collector.get_head(feature_dict, args)

    #VAE
    vae_model = transfer_model.FeatsVAE(args)
    vae_model = vae_model.to(args.device)
    vae_optimizer = Adam(params=filter(lambda p: p.requires_grad, vae_model.parameters()),
                         lr=args.vae_learning_rate)
    if args.pre_vaed== False:
        logger.info('Get VAE Datset')
        start_time = time.time()
        train_vae_loader, valid_vae_loader = transfer_data.get_dataset(feature_dict, head_list, prototype_dict, args)
        logger.info('VAE training')
        transfer_train.train(vae_model, vae_optimizer, train_vae_loader, valid_vae_loader, prototype_dict, args)
        logger.info('Time for VAE: %.1f s' % time_since(start_time))
        logger.info(f'Best VAE Model Path: {args.check_pt_vae_model_path}')
    else:
        args.check_pt_vae_model_path = args.pre_vaed_path

    #Augmentation
    logger.info('Augmentation')
    start_time = time.time()
    vae_model.load_state_dict(torch.load(args.check_pt_vae_model_path, map_location=args.device))
    calibration_loader = generator.generate(vae_model, tail_list, prototype_dict, feature_dict, args)
    logger.info('Time for Augmentation: %.1f s' % time_since(start_time))

    #Calibration
    new_model = Classifier(args)
    if args.pre_calibrated == False:
        logger.info('Calibration')
        start_time = time.time()
        new_optimizer = DenseSparseAdam(new_model.parameters())
        calibration.calibrate(model, new_model, new_optimizer, train_loader, val_loader, test_loader, calibration_loader, mlb, args)
        logger.info('Time for Calibration: %.1f s' % time_since(start_time))
        logger.info(f'Best Model Path: {args.check_pt_new_model_path}')
    else:
        args.check_pt_new_model_path = args.pre_calibrated_path

    logger.info('Predicting')
    model.load_state_dict(torch.load(args.check_pt_model_path, map_location=args.device))
    new_model.load_state_dict(torch.load(args.check_pt_new_model_path, map_location=args.device))
    result = calibration.test(model, new_model, test_loader, mlb, args)
    logger.info(f'Final Test Result: {result}')

if __name__ == '__main__':
    main()
