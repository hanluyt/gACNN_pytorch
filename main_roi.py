# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_roi import *
import argparse
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()  # np.array

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_roi_final_0913.pth" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def setup(args):
    model = ACNN()
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def valid(args, model, test_loader, global_step):
    # Validation
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info(" Num steps = %d", len(test_loader))
    logger.info(" Batch size = %d", args.batchsize)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        imgs, landmarks, targets = batch['image'], batch['landmarks'] / 8. + 6, batch['label']
        landmarks = roi_select(landmarks)
        landmarks = landmarks.float()
        imgs, landmarks, targets = imgs.to(args.device), landmarks.to(args.device), targets.to(args.device)
        with torch.no_grad():
            logits = model(imgs, landmarks)
            eval_loss = loss_fct(logits, targets)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(targets.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], targets.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global epochs: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy, eval_losses.avg

def train(args, model):
    """ Train the model """
    # Prepare dataset
    shuffle = False
    train_set = FaceLandmarksDataset(csv_file='train_aug_24.csv', root_dir='RAF_aug_train/',
                                     transform=transforms.Compose([Rescale((224, 224)), ToTensor()]))
    test_set = FaceLandmarksDataset(csv_file='test_aug_24.csv', root_dir='RAF_aug_test/',
                                    transform=transforms.Compose([Rescale((224, 224)), ToTensor()]))

    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=args.batchsize, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, shuffle=shuffle, batch_size=args.batchsize, num_workers=4, pin_memory=True)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(t_total/10), int(t_total/2)],gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    loss_fct = nn.CrossEntropyLoss()
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_steps)
    logger.info("  batch size = %d", args.batchsize)

    model.zero_grad()
    # set_seed(args)  # Added here for reproducibility
    global_epoch, best_acc = 0, 0
    train_losses = AverageMeter()
    # train_preds, train_label = [], []
    total_train_acc, total_train_loss, total_test_acc, total_test_loss = [], [], [], []
    while True:
        model.train()
        train_preds, train_label = [], []
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X epochs) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            imgs, landmarks, targets = batch['image'], batch['landmarks'] / 8. + 6, batch['label']
            landmarks = roi_select(landmarks)
            landmarks = landmarks.float()
            imgs, landmarks, targets = imgs.to(args.device), landmarks.to(args.device), targets.to(args.device)
            logits = model(imgs, landmarks)
            train_loss = loss_fct(logits, targets)
            train_losses.update(train_loss.item())
            preds = torch.argmax(logits, dim=-1)
            optimizer.zero_grad()
            if len(train_preds) == 0:
                train_preds.append(preds.detach().cpu().numpy())
                train_label.append(targets.detach().cpu().numpy())
            else:
                train_preds[0] = np.append(train_preds[0], preds.detach().cpu().numpy(), axis=0)
                train_label[0] = np.append(train_label[0], targets.detach().cpu().numpy(), axis=0)

            epoch_iterator.set_description(
                "Training (%d / %d epochs) (loss=%2.5f)" % (global_epoch+1, t_total, train_losses.val))
            train_loss.backward()
            scheduler.step()
            optimizer.step()

        # for name, parms in model.named_parameters():
        #     print('->name:', name, '->grad_requires:', parms.requires_grad,
        #           '->grad_value:', parms.grad)

        global_epoch += 1

        train_preds, train_label = train_preds[0], train_label[0]
        train_acc = simple_accuracy(train_preds, train_label)
        train_loss = train_losses.avg
        logger.info("Train Loss: %2.5f" % train_loss)
        logger.info("Train Accuracy: %2.5f" % train_acc)
        total_train_acc.append(train_acc), total_train_loss.append(train_loss)

        test_acc, test_loss = valid(args, model, test_loader, global_epoch)
        total_test_acc.append(test_acc), total_test_loss.append(test_loss)
        if best_acc < test_acc:
            save_model(args, model)
            best_acc = test_acc
            model.train()

        train_losses.reset()
        if global_epoch == t_total:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    train_metric = [[acc, loss] for acc, loss in zip(total_train_acc, total_train_loss)]
    test_metric = [[acc, loss] for acc, loss in zip(total_test_acc, total_test_loss)]
    train_metric = pd.DataFrame(train_metric, columns=['train_acc', 'train_loss'])
    test_metric = pd.DataFrame(test_metric, columns=['test_acc', 'test_loss'])
    train_metric.to_csv('train_acnn_0913.csv', index=False)
    test_metric.to_csv('test_acnn_0913.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--output_dir", default="acnn_result", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--batchsize", default=64, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--weight_decay", default=5e-4, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--num_steps", default=600, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    # args.name = 'acnn'
    # args.batchsize = 8
    # args.learning_rate = 3e-2
    # args.seed = 42
    # args.num_steps = 10
    # args.output_dir = 'acc_result'
    # args.weight_decay = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # Set seed
    # set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
