from config import *
import argparse
import os
import time
import pickle
import random
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import re
from nanopore_dataloader import differences_transform, noise_transform,\
                                startMove_transform, cutToWindows_transform, ReduceLROnPlateau
from nanopore_dataloader_new import NanoporeDataset
from nanopore_models import bnLSTM_32window
from sklearn.model_selection import train_test_split
from train_logger import TrainLogger

matplotlib.use("agg")

torch.manual_seed(1337)
np.random.seed(1337)  # Not sure if this one is necessary
random.seed(1337)

accArryForTest = []
lossArryForTest = []  # TODO: move to logger


class NanoporeTrainer(object):
    def __init__(self, model_name, model_folder, hidden_size, batch_size, max_iter, use_gpu):
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._use_gpu = use_gpu
        self._shuffle_datasets = SHUFFLE_DATA
        self._test_batches = TEST_BATCHES
        self._logger = TrainLogger()
        self._logger.init_metric('trainAccForTest')
        self._logger.init_metric('trainLossForTest')

        self._stride = STRIDE
        self._win_len = WIN_LEN
        self._seq_len = SEQ_LEN
        self._optim_lr = OPTIM_LR
        # self._out_channele = OUT_CHANNELS  # TODO: Deprecated?

        self._train_reads = None
        self._test_reads = None
        self._load_data()

        self._transform = None
        self._train_dataset = None
        self._test_dataset = None
        self._init_datasets()

        self._train_dataloader = None
        self._test_dataloader = None
        self._init_dataloaders()

        self._model = None
        self._model_save_path = None
        self._loss_fn = None
        self._optimizer = None
        self._scheduler = None
        self._load_model(model_name)

        self._log_file = open(os.path.join(MODEL_FOLDER_NAME, SUMMARY_FILENAME), 'w')
        self._log_file.write(LOG_HEADER)
        self._chrom_epoch = 0
        self._mito_epoch = 0

    def _load_metadata_pickle(self, path):
        with open(path, 'rb') as f:
            reads_metadata = pickle.load(f)
        return reads_metadata

    def _load_data(self):
        if not os.path.exists(MODEL_FOLDER_NAME):
            os.makedirs(MODEL_FOLDER_NAME)
        positive_reads = self._load_metadata_pickle(os.path.join(DATA_FOLDER, 'tumor', METADATA_FILENAME))
        negative_reads = self._load_metadata_pickle(os.path.join(DATA_FOLDER, 'non_tumor', METADATA_FILENAME))
        train_pos, test_pos = train_test_split(positive_reads, train_size=.7)
        train_neg, test_neg = train_test_split(negative_reads, train_size=.7)
        self._train_reads = {1: train_pos, 0: train_neg}
        self._test_reads = {1: test_pos, 0: test_neg}
        print("Num train samples: %s" % (len(train_pos) + len(train_neg)))
        print("Num test samples: %s" % (len(test_pos) + len(test_neg)))


    def _init_datasets(self):
        self._transform = transforms.Compose([
            transforms.Lambda(lambda x: startMove_transform(x)),
            transforms.Lambda(lambda x: differences_transform(x)),
            transforms.Lambda(lambda x: cutToWindows_transform(x, self._seq_len, self._stride, self._win_len)),
            transforms.Lambda(lambda x: noise_transform(x)),
        ])
        self._train_dataset = NanoporeDataset(self._train_reads, transform=self._transform)
        self._test_dataset = NanoporeDataset(self._test_reads, transform=self._transform)

    def _init_dataloaders(self):
        self._train_dataloader = DataLoader(dataset=self._train_dataset,
                                               batch_size=self._batch_size,
                                               drop_last=False,
                                               num_workers=NUM_WORKERS, pin_memory=USE_PIN_MEMORY)
        self._init_test_dataloader()

    def _init_test_dataloader(self):
        self._test_dataloader = iter(DataLoader(dataset=self._test_dataset,
                                       batch_size=self._batch_size,
                                       drop_last=False,
                                       num_workers=NUM_WORKERS, pin_memory=USE_PIN_MEMORY))

    def _load_model(self, model_name, load=False):
        ## Load any model from "nanopore_models" and set the parameters as in the example below

        # model = VDCNN_withDropout_normalMaxPool(input_size=self._win_len, self._hidden_size=self._hidden_size,\
        #  max_length=self._seq_len, n_classes=2, depth=9,\
        #  n_fc_neurons=1024, shortcut=True,\
        #  dropout=0.5)
        self._model = bnLSTM_32window(input_size=self._win_len, hidden_size=self._hidden_size, max_length=self._seq_len,
                                num_layers=1, use_bias=True, batch_first=True, dropout=0.5, num_classes=2)
        self._model_save_path = '{}/{}/{}'.format(MODELS_DIR, model_name, "Nanopore_model.pth")
        if load:
            self._model = torch.load(self._model_save_path)

        if self._use_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self._model.cuda()
        self._loss_fn = nn.CrossEntropyLoss().cuda()

        # momentum = 0.5 # TODO: Deprecated?
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._optim_lr, eps=1e-05, weight_decay=0)
        self._scheduler = ReduceLROnPlateau(self._optimizer, 'min')

    def _print_train_stats(self, currentBatchNumber, epoch, train_loss, train_accuracy, percentageCategory, scores):
        for param_group in self._optimizer.param_groups:
            current_lr = float(param_group['lr'])
        print("epoch: ", str(epoch), " | batch number: ", currentBatchNumber, \
              " | start/current LR:", str(self._optim_lr), ",", str(current_lr))
        # " | reads left: ", len(train_chrom_dataset)," out of ", len(wholeData_Chrom))
        print("loss is: ", "{0:.4f}".format( \
            train_loss.data.item()), \
              " \nand acc is: ", "{0:.4f}".format( \
                train_accuracy.data.item()))
        print("acc for classes: ", percentageCategory)
        unique, counts = np.unique(torch.max(scores, 1)[1].data.cpu().numpy(), return_counts=True)
        print("", dict(zip(unique, counts)))

    def train(self):
        currentBatchNumber = 0
        numTotalSamples = 0
        numCorrectPosSamples = 0
        numCorrectNegSamples = 0
        for epoch in tqdm(range(self._max_iter), total=self._max_iter):
            for train_batch, train_labels in self._train_dataloader:
                currentBatchNumber += 1
                train_batch = train_batch.cpu()
                train_labels = train_labels.cpu()
                numTotalSamples += len(train_labels)
                if self._use_gpu:
                    train_batch = train_batch.cuda()
                    train_labels = train_labels.cuda()

                ################### Traning
                self._model.train(True)
                self._model.zero_grad()
                train_loss, train_accuracy, scores = self.compute_loss_accuracy(
                    data=train_batch,
                    label=train_labels,
                    model=self._model, validation=True)
                if currentBatchNumber > 100:
                    self._scheduler.step(sum(self._logger['loss'][-50:]) / 50, currentBatchNumber)
                #
                # if currentBatchNumber > 99:
                #     if sum(lossArray[-20:]) / 20 < minLoss:
                #         minLoss = sum(lossArray[-20:]) / 20
                #         minLossModel = model

                ## logging and printing accuracy and loss
                self._logger.log_metric('loss', train_loss.data.item())
                self._logger.log_metric('acc', train_accuracy.data.item())

                valid_categoryAccCounter = [0 for k in range(2)]
                valid_categorySampleCounter = [0 for k in range(2)]
                for position, k in enumerate(train_labels):
                    valid_categorySampleCounter[k.data.item()] += 1
                    if (k.data.item() == torch.max(scores, 1)[1].data[position]):
                        if k.data.item() == 1:
                            numCorrectPosSamples += 1
                        if k.data.item() == 0:
                            numCorrectNegSamples += 1
                        valid_categoryAccCounter[k.data.item()] += 1
                percentageCategory = (np.divide(valid_categoryAccCounter, valid_categorySampleCounter))
                percentageCategory[np.isnan(percentageCategory)] = 0

                train_loss.backward()
                self._optimizer.step()

                if currentBatchNumber % 20 == 0:
                    self._print_train_stats(currentBatchNumber, epoch, train_loss, train_accuracy, percentageCategory,
                                            scores)
                if currentBatchNumber % 100 == 0:
                    del train_loss, train_accuracy, scores
                    self.validation(epoch, currentBatchNumber)

    def validation(self, epoch, currentBatchNumber):
        lossPlt = plt.figure()
        for test_iteration in range(NUM_TEST_BATCHES):
            try:
                currentBatch, currentBatch_labels = self._test_dataloader.__next__()
            except StopIteration:
                self._init_test_dataloader()
                currentBatch, currentBatch_labels = self._test_dataloader.__next__()
            currentBatch = currentBatch.cpu()
            currentBatch_labels = currentBatch_labels.cpu()
            currentBatch_labels.requires_grad_(False)

            if self._use_gpu:
                currentBatch = currentBatch.cuda()
                currentBatch_labels = currentBatch_labels.cuda()

            ################### Testing
            self._model.train(False)
            valid_loss, valid_accuracy, valid_scores = self.compute_loss_accuracy(
                data=currentBatch,
                label=currentBatch_labels,
                model=self._model, validation=True)

            self._logger.log_metric('testLoss', valid_loss.data.item())
            self._logger.log_metric('testAcc', valid_accuracy.data.item())

            valid_categoryAccCounter = [0 for k in range(2)]
            valid_categorySampleCounter = [0 for k in range(2)]
            for position, k in enumerate(currentBatch_labels):
                valid_categorySampleCounter[k.data.item()] += 1
                if (k.data.item() == torch.max(valid_scores, 1)[1].data[position]):
                    valid_categoryAccCounter[k.data.item()] += 1
            percentageCategory = (np.divide(valid_categoryAccCounter, valid_categorySampleCounter))
            percentageCategory[np.isnan(percentageCategory)] = 0
            self._logger.log_metric('allChromAccTest', np.multiply(percentageCategory, 100).astype(int))

        self._logger['trainAccForTest'].extend(self._logger['acc'][-NUM_TEST_BATCHES:])
        self._logger['trainLossForTest'].extend(self._logger['loss'][-NUM_TEST_BATCHES:])

        print("VALIDATION START ===================")
        print("epoch: ", str(epoch), " | batch number: ", currentBatchNumber)
        # " | reads left: ", len(train_chrom_dataset)," out of ", len(wholeData_Chrom))
        print("loss is: ", "{0:.4f}".format(
            sum(self._logger['testLoss'][-NUM_TEST_BATCHES:]) / NUM_TEST_BATCHES),
              " \nand acc is: ", "{0:.4f}".format(
                sum(self._logger['testAcc'][-NUM_TEST_BATCHES:]) / NUM_TEST_BATCHES))
        unique, counts = np.unique(torch.max(valid_scores, 1)[1].data.cpu().numpy(), return_counts=True)
        print("", dict(zip(unique, counts)))

        filenameEndingString = "_Plot_batchSize" + str(self._batch_size) + "_epoch" + str(epoch)
        if currentBatchNumber > 150:
            os.remove(model_folder + "/" + 'trainLoss' + filenameEndingString + '.pdf')
            os.remove(model_folder + "/" + 'trainAcc' + filenameEndingString + '.pdf')
            os.remove(model_folder + "/" + 'testChromAcc' + filenameEndingString + '.pdf')
            os.remove(model_folder + "/" + 'testLoss' + filenameEndingString + '.pdf')
            os.remove(model_folder + "/" + 'testAcc' + filenameEndingString + '.pdf')

        plt.plot(self._logger['loss'])
        lossPlt.savefig(model_folder + "/" + 'trainLoss' + filenameEndingString + '.pdf')
        plt.clf()
        plt.plot(self._logger['acc'])
        lossPlt.savefig(model_folder + "/" + 'trainAcc' + filenameEndingString + '.pdf')
        plt.clf()
        plt.plot(self._logger['allChromAccTest'])
        lossPlt.savefig(model_folder + "/" + 'testChromAcc' + filenameEndingString + '.pdf')
        plt.clf()
        plt.plot(lossArryForTest)
        plt.plot(self._logger['testLoss'])
        lossPlt.savefig(model_folder + "/" + 'testLoss' + filenameEndingString + '.pdf')
        plt.clf()
        plt.plot(accArryForTest)
        plt.plot(self._logger['testAcc'])
        lossPlt.savefig(model_folder + "/" + 'testAcc' + filenameEndingString + '.pdf')
        plt.clf()

        if currentBatchNumber > 100 and currentBatchNumber % 100 == 0:
            torch.save(self._model, self._model_save_path)
        print("VALIDATION END ===================")
        del valid_loss, valid_accuracy, valid_scores
        self._model.train(True)

    def compute_loss_accuracy(self, data, label, model, validation=False):
        if validation:
            logits = model(input_=data)
            loss = self._loss_fn(input=logits, target=label)
            accuracy = (logits.max(1)[1] == label).float().mean()
            return loss, accuracy, logits

        h_n = model(input_=data)
        h_n2 = model(input_=data)
        logits = h_n
        ## fraternal loss
        kappa_logits = h_n2
        loss = 1 / 2 * (self._loss_fn(logits, label) + self._loss_fn(kappa_logits, label))
        loss = loss + KAPPA * (logits - kappa_logits).pow(2).mean()
        accuracy = (logits.max(1)[1] == label).float().mean()
        return loss, accuracy, logits




### Command example to run DL training
### python nanopore_training.py --hidden-size 512 --batch-size 32 --max-iter 100 --gpu

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the model.')
    parser.add_argument('--hidden-size', required=True, type=int,
                        help='The number of hidden units')
    parser.add_argument('--batch-size', required=True, type=int,
                        help='The size of each batch')
    parser.add_argument('--max-iter', required=True, type=int,
                        help='The maximum iteration count')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='The value specifying whether to use GPU')
    args = parser.parse_args()

    modelFolderPattern = re.compile(".*_winlen.*")
    for model_name in os.listdir(MODELS_DIR):
        if modelFolderPattern.match(model_name):
            model_folder = MODELS_DIR + "/" + model_name
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

    trainer = NanoporeTrainer(model_name, model_folder, hidden_size=args.hidden_size, batch_size=args.batch_size,
                              max_iter=args.max_iter, use_gpu=args.gpu)
    trainer.train()
