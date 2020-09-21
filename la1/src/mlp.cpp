#include "mlp.hpp"
#include <fstream>

// Constructor: Default values for all the parameters
MLP() {}

// DESTRUCTOR: free memory
~MLP() {}

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
void MLP::trainOnline(Dataset* trainDataset) {}

// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MLP::runOnlineBackPropagation(Dataset* trainDataset, Dataset* testDataset, int maxiter, double* errorTrain, double* errorTest)
{
}

// Optional Kaggle: Save the model weights in a textfile
bool MLP::saveWeights(const char* archivo) {}

// Optional Kaggle: Load the model weights from a textfile
bool MLP::readWeights(const char* archivo) {}