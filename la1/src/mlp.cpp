#include "mlp.hpp"
#include "util.h"
#include <fstream>

void MLP::freeMemory();

// Feel all the weights (w) with random numbers between -1 and +1
void MLP::randomWeights();

// Feed the input neurons of the network with a vector passed as an argument
void MLP::feedInputs(double* input);

// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MLP::getOutputs(double* output);

// Make a copy of all the weights (copy w in wCopy)
void MLP::copyWeights();

// Restore a copy of all the weights (copy wCopy in w)
void MLP::restoreWeights();

// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MLP::forwardPropagate();

// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MLP::obtainError(double* target);

// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MLP::backpropagateError(double* target);

// Accumulate the changes produced by one pattern and save them in deltaW
void MLP::accumulateChange();

// Update the network weights, from the first layer to the last one
void MLP::weightAdjustment();

// Print the network, i.e. all the weight matrices
void MLP::printNetwork();

// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void performEpochOnline(double* input, double* target);

// Constructor: Default values for all the parameters
MLP::MLP() {}

// DESTRUCTOR: free memory
MLP::~MLP() {}

// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MLP::initialize(int nl, int npl[])
{
    for (uint i = 0; i < nl; ++i)
        _layers[i] = *new Layer(npl[i]);
}

// Read a dataset from a file name and return it
Dataset* MLP::readData(const char* fileName)
{
    Dataset* dataset;
    std::ifstream& dataFile;

    dataFile.open(fileName, std::ios::in);
    dataFile >> dataset->nOfInputs >> dataset->nOfOutputs >> dataset->nOfPatterns;
    dataset->inputs = new double*[dataset->nOfPatterns];

    for (uint patternIndex = 0; patternIndex < dataset->nOfPatterns; ++patternIndex) {
        dataset->inputs[patternIndex] = new double[dataset->nOfInputs];
        dataset->outputs[patternIndex] = new double[dataset->nOfOutputs];

        for (uint inputIndex = 0; inputIndex < dataset->nOfInputs; ++inputIndex)
            dataFile >> dataset->inputs[patternIndex][inputIndex];

        for (uint outputIndex = 0; outputIndex < dataset->nOfOutputs; ++outputIndex)
            dataFile >> dataset->inputs[patternIndex][outputIndex];
    }

    dataFile.close();

    return dataset;
}

// Test the network with a dataset and return the MSE
double MLP::test(Dataset* dataset) {}

// Obtain the predicted outputs for a dataset
void MLP::predict(Dataset* testDataset) {}

// Perform an online training for a specific dataset
void MLP::trainOnline(Dataset* trainDataset)
{
    int* permutation = util::integerRandomVectoWithoutRepeating(0, trainDataset->nOfPatterns - 1, trainDataset->nOfPatterns);
    for (uint i = 0; i < trainDataset->nOfPatterns; ++i) {
        double* inputs = trainDataset->inputs[i];
        double* outputs = trainDataset->outputs[i];
        performEpochOnline(inputs, outputs);
    }
}

// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MLP::runOnlineBackPropagation(Dataset* trainDataset, Dataset* testDataset, int maxiter, double* errorTrain, double* errorTest)
{
    errorTrain = new double[trainDataset->nOfPatterns];
    for (uint iter = 0; iter < maxiter; ++iter) {
        trainOnline(testDataset);
    }
    *errorTrain = test(trainDataset);
    *errorTest = test(testDataset);
}

// Optional Kaggle: Save the model weights in a textfile
bool MLP::saveWeights(const char* archivo) {}

// Optional Kaggle: Load the model weights from a textfile
bool MLP::readWeights(const char* archivo) {}