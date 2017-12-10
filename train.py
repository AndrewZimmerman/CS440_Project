import numpy as np
import neuralnetworks as nn
import mlutils as ml #for partition method
import time #used to time each repetition
import csv#used for importing data sets
import matplotlib.figure

def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify = False):
    results = []
    for structure in hiddenLayerStructures:
        print(structure, end=" ")
        #time each hidden layer structure
        start_time = time.time()
        structureData = [structure]
        trainDataResults = []
        testDataResults = []
        for i in range(0,numberRepetitions):
            #partition data
            Xtrain,Ttrain,Xtest,Ttest = ml.partition(X,T, (trainFraction, 1-trainFraction),classification=classify)
            if not classify:
                #create/train network
                nnet = nn.NeuralNetwork(Xtrain.shape[1], structure, Ttrain.shape[1])
                nnet.train(Xtrain, Ttrain, nIterations=numberIterations)
                #test netork
                Ytrain = nnet.use(Xtrain)
                Ytest = nnet.use(Xtest)
                #add error for testing and traing data
                trainDataResults.append(np.sqrt(np.mean((Ytrain-Ttrain)**2)))
                testDataResults.append(np.sqrt(np.mean((Ytest-Ttest)**2)))
            else:
                #create/train network
                nnet = nn.NeuralNetworkClassifier(Xtrain.shape[1],structure,np.unique(Ttrain).size)
                nnet.train(Xtrain,Ttrain,nIterations=numberIterations)
                #test netork
                Ptrain = nnet.use(Xtrain)
                Ptest = nnet.use(Xtest)
                #add error for testing and traing data
                trainDataResults.append(1-(np.sum(Ptrain==Ttrain)/len(Ttrain)))
                testDataResults.append(1-(np.sum(Ptest==Ttest)/len(Ttest)))
        structureData.append(trainDataResults)
        structureData.append(testDataResults)
        structureData.append(time.time() - start_time)
        results.append(structureData)
        print("done")
    return results

def trainNNsWithTrainData(Xtrain, Ttrain, Xtest,Ttest, hiddenLayerStructures, numberRepetitions, numberIterations, classify = False):
    results = []
    for structure in hiddenLayerStructures:
        print(structure, end=" ")
        #time each hidden layer structure
        start_time = time.time()
        structureData = [structure]
        trainDataResults = []
        testDataResults = []
        for i in range(0,numberRepetitions):
            if not classify:
                #create/train network
                nnet = nn.NeuralNetwork(Xtrain.shape[1], structure, Ttrain.shape[1])
                nnet.train(Xtrain, Ttrain, nIterations=numberIterations)
                #test netork
                Ytrain = nnet.use(Xtrain)
                Ytest = nnet.use(Xtest)
                #add error for testing and traing data
                trainDataResults.append(np.sqrt(np.mean((Ytrain-Ttrain)**2)))
                testDataResults.append(np.sqrt(np.mean((Ytest-Ttest)**2)))
            else:
                #create/train network
                nnet = nn.NeuralNetworkClassifier(Xtrain.shape[1],structure,np.unique(Ttrain).size)
                nnet.train(Xtrain,Ttrain,nIterations=numberIterations)
                #test netork
                Ptrain = nnet.use(Xtrain)
                Ptest = nnet.use(Xtest)
                #add error for testing and traing data
                trainDataResults.append(1-(np.sum(Ptrain==Ttrain)/len(Ttrain)))
                testDataResults.append(1-(np.sum(Ptest==Ttest)/len(Ttest)))
        structureData.append(trainDataResults)
        structureData.append(testDataResults)
        structureData.append(time.time() - start_time)
        results.append(structureData)
        print("done")
    return results


def summarize(results):
    summaryData = []
    for structureData in results:
        structureSummaryData = [structureData[0]]
        structureSummaryData.append(np.mean(structureData[1]))
        structureSummaryData.append(np.mean(structureData[2]))
        structureSummaryData.append(structureData[3])
        summaryData.append(structureSummaryData)
    return summaryData

def bestNetwork(summary):
    bestIndex = 0
    bestValue = summary[0][2]
    index = 0
    for structureData in summary:
        if structureData[2] < bestValue:
            bestValue = structureData[2]
            bestIndex = index
        index+=1
    return summary[bestIndex]
