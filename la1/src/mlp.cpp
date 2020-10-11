#include "mlp.hpp"
#include "util.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

using std::vector;
using namespace imc;

void MLP::_freeMemory()
{
}

void MLP::_clearDeltas()
{
    for (Layer& layer : _layers)
        layer.clearDeltas();
}

void MLP::_saveDeltas()
{
    for (Layer& layer : _layers)
        layer.saveDeltas();
}

// Fill all the weights (w) with random numbers between -1 and +1
void MLP::_randomWeights()
{
    uint inputsPerNeuron = 1;

    for (Layer& layer : _layers) {
        layer.randomWeights(inputsPerNeuron);
        inputsPerNeuron = layer.numberOfNeurons();
    }
}

// Feed the input neurons of the network with a vector passed as an argument
void MLP::_feedInputs(double* input)
{
    vector<double> inputVector;
    for (int i = 0; i < _layers[0].numberOfNeurons(); ++i)
        inputVector.push_back(input[i]);

    _layers[0].feed(inputVector, true);
}

// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MLP::_getOutputs(double** output)
{
    *output = _outputPointer();
}

// Make a copy of all the weights (copy w in wCopy)
void MLP::_copyWeights()
{
    for (Layer& layer : _layers)
        layer.copyWeights();
}

// Restore a copy of all the weights (copy wCopy in w)
void MLP::_restoreWeights()
{
    for (Layer& layer : _layers)
        layer.restoreWeights();
}

// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MLP::_forwardPropagate()
{
    for (uint i = 1; i < _layers.size(); ++i)
        _layers[i].feed(_layers[i - 1].out());
}

// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MLP::_obtainError(double* target)
{
    double squaredError = .0;
    double currentError;
    double* output;
    _getOutputs(&output);
    for (uint i = 0; i < _lastLayer().numberOfNeurons(); ++i) {
        currentError = target[i] - output[i];
        squaredError += pow(currentError, 2);
    }
    return squaredError / _lastLayer().numberOfNeurons();
}

// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MLP::_backpropagateError(double* target)
{
    double outError;

    _lastLayer().backpropagate(target);

    for (int i = _layers.size() - 2; i > 0; --i) {
        _layers[i].backpropagate(_layers[i + 1]);
    }
}

// Accumulate the changes produced by one pattern and save them in deltaW
void MLP::_accumulateChange()
{
    for (uint i = 1; i < _layers.size(); ++i)
        _layers[i].accumulateChange();
}

// Update the network weights, from the first layer to the last one
void MLP::_weightAdjustment()
{
    for (uint i = 1; i < _layers.size(); ++i) {
        int H = _layers.size() - 1, h = (int)i;
        double decrement = pow(decrementFactor, h - H);
        _layers[i].weightAdjustement(eta * decrement, mu);
    }
}

// Print the network, i.e. all the weight matrices
void MLP::_printNetwork()
{
    std::cout << "NETWORK WEIGHTS\n===============\n";
    for (int i = 1; i < _layers.size(); ++i) {
        std::cout << "Layer " << i << "\n-------" << std::endl;
        _layers[i].printMatrix();
    }
}

double* MLP::_outputPointer()
{
    double* result = new double[_lastLayer().numberOfNeurons()];

    for (int i = 0; i < _lastLayer().numberOfNeurons(); ++i)
        result[i] = _lastLayer().out()[i];

    return result;
}

// Constructor: Default values for all the parameters
MLP::MLP()
{
    eta = 0.1;
    mu = 0.9;
    validationRatio = 0;
    decrementFactor = 1;
}

// DESTRUCTOR: free memory
MLP::~MLP() { }

// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
int MLP::initialize(int nl, int npl[])
{
    _layers = *new vector<Layer>(nl);

    // This should be corrected
    _layers[0] = *new Layer(npl[0], 1);

    for (uint i = 1; i < nl - 1; ++i)
        _layers[i] = *new Layer(npl[1], _layers[i - 1].numberOfNeurons());

    _layers[nl - 1] = *new Layer(npl[2], _layers[nl - 2].numberOfNeurons());
}

// Read a dataset from a file name and return it
Dataset* MLP::readData(const char* fileName)
{
    Dataset* dataset = new Dataset;
    std::ifstream dataFile(fileName, std::ifstream::in);
    if (!dataFile) {
        std::cerr << "Error opening file " << fileName << ". Exiting.\n";
        exit(EXIT_FAILURE);
    }

    dataFile >> dataset->nOfInputs;
    dataFile >> dataset->nOfOutputs;
    dataFile >> dataset->nOfPatterns;
    dataset->inputs = new double*[dataset->nOfPatterns];
    dataset->outputs = new double*[dataset->nOfPatterns];

    for (uint patternIndex = 0; patternIndex < dataset->nOfPatterns; ++patternIndex) {
        dataset->inputs[patternIndex] = new double[dataset->nOfInputs];
        dataset->outputs[patternIndex] = new double[dataset->nOfOutputs];

        for (uint inputIndex = 0; inputIndex < dataset->nOfInputs; ++inputIndex)
            dataFile >> dataset->inputs[patternIndex][inputIndex];

        for (uint outputIndex = 0; outputIndex < dataset->nOfOutputs; ++outputIndex)
            dataFile >> dataset->outputs[patternIndex][outputIndex];
    }

    dataFile.close();

    return dataset;
}

// Test the network with a dataset and return the MSE
double MLP::test(Dataset* dataset)
{
    double accumulatedError = .0, meanError;

    for (uint patternIndex = 0; patternIndex < dataset->nOfPatterns; ++patternIndex) {
        _feedInputs(dataset->inputs[patternIndex]);
        _forwardPropagate();
        accumulatedError += _obtainError(dataset->outputs[patternIndex]);
    }

    meanError = accumulatedError / dataset->nOfPatterns;
    return meanError;
}

Dataset* MLP::_datasetFromIndexes(Dataset* dataset, int* indexes, int size)
{
    Dataset* result = new Dataset;

    result->inputs = new double*[size];
    result->outputs = new double*[size];
    result->nOfInputs = dataset->nOfInputs;
    result->nOfOutputs = dataset->nOfOutputs;
    result->nOfPatterns = size;

    for (uint i = 0; i < size; ++i) {
        result->inputs[i] = *new double*(dataset->inputs[indexes[i]]);
        result->outputs[i] = *new double*(dataset->outputs[indexes[i]]);
    }
    return result;
}

void* MLP::_splitDataset(Dataset* dataset, Dataset** train, Dataset** validation)
{
    int valSize = validationRatio * dataset->nOfPatterns;
    int trainSize = dataset->nOfPatterns - valSize;
    int* trainingIndexes;
    int* validationIndexes = util::integerRandomVectoWithoutRepeating(0, dataset->nOfPatterns - 1, valSize, &trainingIndexes);

    *train = _datasetFromIndexes(dataset, trainingIndexes, trainSize);
    *validation = _datasetFromIndexes(dataset, validationIndexes, valSize);
}

void MLP::_predictPretty(Dataset* dataset, const unsigned int& patternIndex)
{
    double* predictedOutputs = new double[dataset->nOfOutputs];
    _feedInputs(dataset->inputs[patternIndex]);
    _forwardPropagate();
    _getOutputs(&predictedOutputs);

    for (unsigned int i = 0; i < dataset->nOfOutputs; ++i)
        std::cout << dataset->outputs[patternIndex][i] << " ";

    std::cout << "-- ";

    for (unsigned int i = 0; i < dataset->nOfOutputs; ++i)
        std::cout << predictedOutputs[i] << " ";
}

void MLP::_predict(Dataset* dataset, const unsigned int& patternIndex)
{
    double* predictedOutputs = new double[dataset->nOfOutputs];
    _feedInputs(dataset->inputs[patternIndex]);
    _forwardPropagate();
    _getOutputs(&predictedOutputs);
    std::cout << patternIndex;

    for (unsigned int i = 0; i < dataset->nOfOutputs; ++i)
        std::cout << "," << predictedOutputs[i];
    std::cout << "\n";
}

void MLP::_checkEarlyStopping(const double deltaTrainError, const double deltaValidationError, int& itersNoTrainIncrease, int& itersNoValIncrease)
{
    if (deltaTrainError < 0.000001)
        itersNoTrainIncrease++;
    else
        itersNoTrainIncrease = 0;

    if (this->validationRatio > 0.00001) {
        if (deltaValidationError < 0.000001)
            itersNoValIncrease++;
        else
            itersNoValIncrease = 0;
    }
}

void MLP::_performEpochOnline(double* input, double* target)
{
    _clearDeltas();
    _feedInputs(input);
    _forwardPropagate();
    _backpropagateError(target);
    _accumulateChange();
    _weightAdjustment();
    _saveDeltas();
}

// Obtain the predicted outputs for a dataset
void MLP::predictPretty(Dataset* testDataset)
{
    std::cout << "Desired output Vs Expected output (test)\n========================================" << std::endl;

    for (uint patternIndex = 0; patternIndex < testDataset->nOfPatterns; ++patternIndex) {
        _predictPretty(testDataset, patternIndex);
    }
}

void MLP::predict(Dataset* testDataset)
{
    std::cout << "Id,Predicted\n";
    for (uint patternIndex = 0; patternIndex < testDataset->nOfPatterns; ++patternIndex) {
        _predict(testDataset, patternIndex);
    }
}

// Perform an online training for a specific dataset
void MLP::trainOnline(Dataset* trainDataset)
{
    for (uint i = 0; i < trainDataset->nOfPatterns; ++i)
        _performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
}

// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MLP::runOnlineBackPropagation(Dataset* trainDataset, Dataset* testDataset, int maxiter, double* errorTrain, double* errorTest)
{
    double currTrainError = 1.0, bestError = 1.0, currValError = 0, prevTrainError = 1.0, prevValError = 1.0;
    int iteration = 0, itersNoTrainIncrease = 0, itersNoValIncrease = 0;

    Dataset *train, *validation;

    if (this->validationRatio > 0.00001) {
        _splitDataset(trainDataset, &train, &validation);
        currValError = 1;
    } else
        train = trainDataset;

    _randomWeights();

    while ((iteration < maxiter) && (itersNoTrainIncrease < 100) && (itersNoValIncrease < 50)) {
        iteration++;

        trainOnline(train);

        prevTrainError = currTrainError;
        currTrainError = test(train);

        if (this->validationRatio > 0.00001) {
            prevValError = currValError;
            currValError = test(validation);
        }

        if (currTrainError < bestError) {
            bestError = currTrainError;
            _copyWeights();
        }

        //std::cout << "Iteration " << iteration << "  Training error: " << currTrainError << "  Validation error: " << currValError << '\r' << std::flush;
        std::cout << "Iteration " << iteration << "  Training error: " << currTrainError << "  Validation error: " << currValError << '\n';
        _checkEarlyStopping(prevTrainError - currTrainError, prevValError - currValError, itersNoTrainIncrease, itersNoValIncrease);
    }

    _restoreWeights();
    _printNetwork();
    //predict(testDataset);

    *errorTrain = test(trainDataset);
    *errorTest = test(testDataset);
}

// Optional Kaggle: Save the model weights in a textfile
bool MLP::saveWeights(const char* archivo)
{
    std::ofstream file(archivo, std::ios::out);
    file << _layers[0].numberOfNeurons() << " " << _layers.size() - 2 << " " << _layers[_layers.size() - 1].numberOfNeurons() << std::endl;
    for (int i = 1; i < _layers.size(); ++i) {
        file << _layers[i].numberOfNeurons() << std::endl;
        _layers[i].printMatrix(file);
    }

    file.close();
}

// Optional Kaggle: Load the model weights from a textfile
bool MLP::readWeights(const char* archivo)
{
    int nInputs, hiddenLayers, nOutputs, nLayers;
    int nOfNeurons;
    std::ifstream file(archivo, std::ios::in);
    if (!file) {
        std::cerr << "No se encontró el archivo de pesos. Saliendo." << std::endl;
        exit(EXIT_FAILURE);
    }

    file >> nInputs;
    file >> hiddenLayers;
    file >> nOutputs;

    nLayers = hiddenLayers + 2;
    _layers = *new std::vector<Layer>(nLayers);

    _layers[0] = *new Layer(nInputs, 1);

    for (uint i = 1; i <= hiddenLayers; ++i) {
        file >> nOfNeurons;
        _layers[i] = *new Layer(nOfNeurons, _layers[i - 1].numberOfNeurons());

        for (Neuron& neuron : _layers[i].neurons())
            neuron.readWeights(file, _layers[i - 1].numberOfNeurons());
    }

    _layers[nLayers - 1] = *new Layer(nOutputs, _layers[hiddenLayers].numberOfNeurons());

    for (Neuron& neuron : _layers[nLayers - 1].neurons())
        neuron.readWeights(file, _layers[hiddenLayers].numberOfNeurons());
}