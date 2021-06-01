import argparse
import os
import time
import numpy as np
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import re
from nanopore_dataloader import differences_transform, noise_transform,\
                                startMove_transform, cutToWindows_transform, RangeNormalize, \
                                ReduceLROnPlateau
from nanopore_dataloader_new import NanoporeDataset
from nanopore_models import bnLSTM, bnLSTM_32window, regGru_32window_hidden_BN,\
                            simpleCNN_10Layers_noDilation_largeKernel_withDropout,\
                            VDCNN_gru_1window_lastStep
matplotlib.use("agg")





### Command example to run DL training
### python nanopore_training.py --hidden-size 512 --batch-size 32 --max-iter 100 --gpu

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    ## setting parameters
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    max_iter = args.max_iter
    use_gpu = args.gpu
    shuffleDatasets = True
    # batchesForTest - number of times test dataset is reshuffled and rechecked
    batchesForTest = 3


    stride = 1
    winLength = 1
    seqLength = 2000
    outChannele = 64



    ## Loading training data and setting logging location
    torch.manual_seed(7)
    folderWithDataFiles = "./Data/Hek1_example_data/"
    currentModelFolder = "results_onHek"
    searchModelsInFolder = "Models/"
    if not os.path.exists(currentModelFolder):
        os.makedirs(currentModelFolder)


    import pickle
    with open('Data/cancer_sample_data/tumor/test_signal_mapping_tuple.pkl', 'rb') as f:
        all_data_tumor = pickle.load(f)
        tot_samples_tumor = len(all_data_tumor)
    with open('Data/cancer_sample_data/non_tumor/test_signal_mapping_tuple.pkl', 'rb') as f:
        all_data_non = pickle.load(f)
    all_data_non = all_data_non[:len(all_data_tumor)]
    tot_samples_non = len(all_data_non)
    train_data = {1: all_data_tumor[:int(tot_samples_tumor*0.7)], 0: all_data_non[:int(tot_samples_non*0.7)]}
    test_data = {1: all_data_tumor[int(tot_samples_tumor*0.7):], 0: all_data_non[int(tot_samples_non*0.7):]}
    print("Num train samples:")
    print(sum(len(train_data[label]) for label in train_data))
    print("Num test samples:")
    print(sum(len(test_data[label]) for label in test_data))


    ## creating Datasets

    transform = transforms.Compose([
        transforms.Lambda(lambda x: startMove_transform(x)),
        transforms.Lambda(lambda x: differences_transform(x)),
        transforms.Lambda(lambda x: cutToWindows_transform(x, seqLength, stride, winLength)),
        transforms.Lambda(lambda x: noise_transform(x)),
    ])

    train_dataset = NanoporeDataset(train_data, transform = transform)
    test_dataset = NanoporeDataset(test_data, transform = transform)


    summaryFile = open(str(currentModelFolder)+'/summaryFile_Cliveome.csv', 'w')
    summaryFile.write('Name, Mito_Total_Reads, Mito_Correct_Reads, Chrom_Total_Reads, Chrom_Correct_Reads \n')
    modelFolderPattern = re.compile(".*_winlen.*")
    for currentTestingModelFolder in os.listdir(searchModelsInFolder):
        if modelFolderPattern.match(currentTestingModelFolder):

            newFinetuneFolder = currentModelFolder + "/" + currentTestingModelFolder
            if not os.path.exists(newFinetuneFolder):
                os.makedirs(newFinetuneFolder)


            print("working folder is: ", currentTestingModelFolder)
            splitModelFolder = currentTestingModelFolder.split("_")
            for splitWord in splitModelFolder:
                print(splitWord)
                if "winlen" in splitWord:
                    currentWinLen = int(splitWord.replace("winlen", ""))
                    if currentWinLen == 1:
                        stride = 1
                        winLength = 1
                        seqLength = 2000
                    elif currentWinLen == 32:
                        stride = 10
                        winLength = 32
                        seqLength = 200




            num_of_workers = 0
            use_pin_memmory = False
            train_dataloader = iter(DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffleDatasets,
                                               drop_last=False,
                                               num_workers=num_of_workers, pin_memory=use_pin_memmory))
            test_dataloader = iter(DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffleDatasets,
                                              drop_last=False,
                                              num_workers=num_of_workers, pin_memory=use_pin_memmory))

            ## Load any model from "nanopore_models" and set the parameters as in the example below

            # model = VDCNN_withDropout_normalMaxPool(input_size=winLength, hidden_size=hidden_size,\
            #  max_length=seqLength, n_classes=2, depth=9,\
            #  n_fc_neurons=1024, shortcut=True,\
            #  dropout=0.5)
            model = bnLSTM_32window(input_size=winLength, hidden_size=hidden_size, max_length=seqLength, num_layers=1,
                                    use_bias=True, batch_first=True, dropout=0.5, num_classes=2)


            saved_model_path = '{}/{}/{}'.format(searchModelsInFolder,currentTestingModelFolder, "Nanopore_model.pth")

            ## or uncomment to load previously saved model located in saved_model_path

            # model= torch.load(saved_model_path)

            if use_gpu:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                model.cuda()

            optim_lr = 0.001
            loss_fn = nn.CrossEntropyLoss().cuda()

            kappa = 0.01
            momentum = 0.5
            optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, eps=1e-05 ,weight_decay=0)
            scheduler = ReduceLROnPlateau(optimizer, 'min')


            ## setting arrays for logging accuracies/losses
            lossArray = []
            accArray = []
            ### lossArryForTest is required in order to plot one graph of train vs valid loss
            lossArryForTest = []
            accArryForTest = []
            lossArrayTest = []
            accArrayTest = []
            allChromAcc = []
            allChromAccTest = []
            minLoss = 999
            maxtestAcc= 0
            minLossModel = None
            matplotlib.interactive(False)
            lossPlt = plt.figure()


            def compute_loss_accuracy(data, label,model = model ,validation = False):
                if validation:
                    logits = model(input_=data)
                    loss = loss_fn(input=logits, target=label)
                    accuracy = (logits.max(1)[1] == label).float().mean()
                    return loss, accuracy, logits

                h_n = model(input_=data)
                h_n2 = model(input_=data)
                logits = h_n
                ## fraternal loss
                kappa_logits = h_n2
                loss = 1/2*(loss_fn(logits, label) + loss_fn(kappa_logits, label))
                loss = loss + kappa * (logits - kappa_logits).pow(2).mean()
                accuracy = (logits.max(1)[1] == label).float().mean()
                return loss, accuracy, logits


            ## setting variables to compute timings
            currentBatchNumber = 0
            numTotalSamples = 0
            numCorrectMitoSamples = 0
            numCorrectChromSamples = 0
            epoch = 0
            while epoch < max_iter:
                epoch += 1
                for train_batch, train_labels in train_dataloader:
                    currentBatchNumber += 1

                    train_batch = train_batch.cpu()
                    train_labels = train_labels.cpu()

                    numTotalSamples += len(train_labels)

                    if use_gpu:
                        train_batch = train_batch.cuda()
                        train_labels = train_labels.cuda()

                    ################### Traning
                    model.train(True)
                    model.zero_grad()
                    train_loss, train_accuracy, scores = compute_loss_accuracy( \
                        data=train_batch,
                        label=train_labels,
                        model=model, validation=True)
                    if currentBatchNumber > 100:
                        scheduler.step(sum(lossArray[-50:])/50, currentBatchNumber)

                    if currentBatchNumber > 99:
                        if sum(lossArray[-20:])/20 < minLoss:
                            minLoss = sum(lossArray[-20:])/20
                            minLossModel = model

                    ## logging and printing accuracy and loss
                    lossArray.append(train_loss.data.item())
                    accArray.append(train_accuracy.data.item())

                    valid_categoryAccCounter = [0 for k in range(2)]
                    valid_categorySampleCounter = [0 for k in range(2)]
                    for position, k in enumerate(train_labels):
                        valid_categorySampleCounter[k.data.item()] += 1
                        if (k.data.item() == torch.max(scores, 1)[1].data[position]):
                            if k.data.item() == 1:
                                numCorrectMitoSamples += 1
                            if k.data.item() == 0:
                                numCorrectChromSamples += 1
                            valid_categoryAccCounter[k.data.item()] += 1
                    percentageCategory = (np.divide(valid_categoryAccCounter, valid_categorySampleCounter))
                    percentageCategory[np.isnan(percentageCategory)] = 0


                    train_loss.backward()
                    optimizer.step()

                    # continue #########################################################
                    if currentBatchNumber % 20 == 0:
                        for param_group in optimizer.param_groups:
                            current_lr = float(param_group['lr'])
                        print("epoch: " , str(epoch), " | batch number: ", currentBatchNumber, \
                              " | start/current LR:", str(optim_lr),",", str(current_lr))
                        # " | reads left: ", len(train_chrom_dataset)," out of ", len(wholeData_Chrom))
                        print("loss is: ", "{0:.4f}".format( \
                            train_loss.data.item()) , \
                              " \nand acc is: ", "{0:.4f}".format( \
                                train_accuracy.data.item() ))
                        print("acc for classes: ",percentageCategory)
                        unique, counts = np.unique(torch.max(scores, 1)[1].data.cpu().numpy(), return_counts=True)
                        print("", dict(zip(unique, counts)))

                    ## running validation
                    if currentBatchNumber % 100 == 0:
                        del train_loss, train_accuracy, scores
                        for test_iteration in range(batchesForTest):
                            try:
                                currentBatch, currentBatch_labels = test_dataloader.__next__()
                            except StopIteration:
                                iter(DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffleDatasets,
                                                drop_last=False,
                                                num_workers=num_of_workers, pin_memory=use_pin_memmory))
                                currentBatch, currentBatch_labels = test_dataloader.__next__()


                            currentBatch = currentBatch.cpu()
                            currentBatch_labels = currentBatch_labels.cpu()
                            currentBatch_labels.requires_grad_(False)

                            if use_gpu:
                                currentBatch = currentBatch.cuda()
                                currentBatch_labels = currentBatch_labels.cuda()

                            ################### Testing
                            model.train(False)
                            valid_loss, valid_accuracy, valid_scores = compute_loss_accuracy(
                                data=currentBatch,
                                label=currentBatch_labels,
                                model=model, validation=True)

                            lossArrayTest.append(valid_loss.data.item())
                            accArrayTest.append(valid_accuracy.data.item())

                            valid_categoryAccCounter = [0 for k in range(2)]
                            valid_categorySampleCounter = [0 for k in range(2)]
                            for position, k in enumerate(currentBatch_labels):
                                valid_categorySampleCounter[k.data.item()] += 1
                                if (k.data.item() == torch.max(valid_scores, 1)[1].data[position]):
                                    valid_categoryAccCounter[k.data.item()] += 1
                            percentageCategory = (np.divide(valid_categoryAccCounter, valid_categorySampleCounter))
                            percentageCategory[np.isnan(percentageCategory)] = 0
                            allChromAccTest.append(np.multiply(percentageCategory, 100).astype(int))


                        accArryForTest = accArryForTest+accArray[-batchesForTest:]
                        lossArryForTest = lossArryForTest+lossArray[-batchesForTest:]



                        print("VALIDATION START ===================")
                        print("epoch: " , str(epoch), " | batch number: ", currentBatchNumber)
                        # " | reads left: ", len(train_chrom_dataset)," out of ", len(wholeData_Chrom))
                        print("loss is: ", "{0:.4f}".format(
                            sum(lossArrayTest[-batchesForTest:])/batchesForTest) ,
                              " \nand acc is: ", "{0:.4f}".format(
                                sum(accArrayTest[-batchesForTest:])/batchesForTest ))
                        unique, counts = np.unique(torch.max(valid_scores, 1)[1].data.cpu().numpy(), return_counts=True)
                        print("", dict(zip(unique, counts)))

                        if currentBatchNumber > 150:
                            os.remove(newFinetuneFolder +"/"+'trainLoss'+filenameEndingString+'.pdf')
                            os.remove(newFinetuneFolder +"/"+'trainAcc'+filenameEndingString+'.pdf')
                            os.remove(newFinetuneFolder +"/"+'testChromAcc'+filenameEndingString+'.pdf')
                            os.remove(newFinetuneFolder +"/"+'testLoss'+filenameEndingString+'.pdf')
                            os.remove(newFinetuneFolder +"/"+'testAcc'+filenameEndingString+'.pdf')


                        filenameEndingString = "_Plot_batchSize"+str(batch_size)+"_epoch"+str(epoch)
                        plt.plot(lossArray)
                        lossPlt.savefig(newFinetuneFolder +"/"+'trainLoss'+filenameEndingString+'.pdf')
                        plt.clf()
                        # lossPlt = plt.figure()
                        plt.plot(accArray)
                        lossPlt.savefig(newFinetuneFolder +"/"+'trainAcc'+filenameEndingString+'.pdf')
                        plt.clf()
                        # lossPlt = plt.figure()
                        plt.plot(allChromAccTest)
                        lossPlt.savefig(newFinetuneFolder +"/"+'testChromAcc'+filenameEndingString+'.pdf')
                        plt.clf()
                        # lossPlt = plt.figure()
                        plt.plot(lossArryForTest)
                        plt.plot(lossArrayTest)
                        lossPlt.savefig(newFinetuneFolder +"/"+'testLoss'+filenameEndingString+'.pdf')
                        plt.clf()
                        plt.plot(accArryForTest)
                        plt.plot(accArrayTest)
                        lossPlt.savefig(newFinetuneFolder +"/"+'testAcc'+filenameEndingString+'.pdf')
                        plt.clf()


                        if currentBatchNumber > 100 and currentBatchNumber % 100 == 0:
                            save_path = '{}/{}'.format(newFinetuneFolder, "Nanopore_model.pth")
                            save_path_notMin = '{}/{}'.format(newFinetuneFolder, "Nanopore_model_notMin.pth")
                            torch.save(minLossModel, save_path)
                        print("VALIDATION END ===================")
                        del valid_loss, valid_accuracy, valid_scores
                        model.train(True)
                    ## end of training and validation


            # print(numTotalMitoSamples ,
            # numTotalChromSamples ,
            # numCorrectMitoSamples ,
            # numCorrectChromSamples )

            # summaryFile.write(str(currentTestingModelFolder) + ", " +str(numTotalMitoSamples) + ", " +str(numCorrectMitoSamples) + ", " +str(numTotalChromSamples) + ", "+ str(numCorrectChromSamples)+',\n')


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
    main()
