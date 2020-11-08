/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <sstream>
#include <string>

using namespace imc;
using namespace std;
using namespace util;

int randomInt(int Low, int High)
{
    int rangeSize = (High - Low) + 1;
    return (rand() % rangeSize) + Low;
}

double randomDouble(double Low, double High)
{
    double randSample = (double)rand() / RAND_MAX;
    return Low + randSample * (High - Low);
}

double MultilayerPerceptron::sigmoid(const double x)
{
    return 1 / (1 + exp(x));
}

int MultilayerPerceptron::getMaxOutput()
{
    int maxIndex = 0, nOfOutputs = layers[nOfLayers - 1].nOfNeurons;
    for (int i = 0; i < nOfOutputs; i++)
        if (layers[nOfLayers - 1].neurons[i].out >= layers[nOfLayers - 1].neurons[maxIndex].out)
            maxIndex = i;
    return maxIndex;
}

MultilayerPerceptron::MultilayerPerceptron()
{
    eta = .1;
    mu = .9;
    validationRatio = 0;
    decrementFactor = 1;
    online = false;
    outputFunction = 0;
}

int MultilayerPerceptron::initialize(int nl, int npl[])
{
    layers = new Layer[nl];

    nOfLayers = nl;

    for (int i = 0; i < nl; ++i) {

        layers[i] = *new Layer;
        layers[i].neurons = new Neuron[npl[i]];
        layers[i].nOfNeurons = npl[i];

        if (i != 0) {
            int inputsPerNeuron = npl[i - 1] + 1;
            for (int j = 0; j < layers[i].nOfNeurons; ++j) {
                Neuron* neuron = layers[i].neurons + j;

                neuron->w = new double[inputsPerNeuron];
                neuron->deltaW = new double[inputsPerNeuron];
                neuron->wCopy = new double[inputsPerNeuron];
                neuron->lastDeltaW = new double[inputsPerNeuron];
            }
        }
    }
}

MultilayerPerceptron::~MultilayerPerceptron()
{
    freeMemory();
}

void MultilayerPerceptron::freeMemory()
{
    delete[] layers;
}

void MultilayerPerceptron::randomWeights()
{
    int inputsPerNeuron = 1;

    for (int i = 1; i < nOfLayers; ++i) {
        inputsPerNeuron = layers[i - 1].nOfNeurons + 1;
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            Neuron* neuron = layers[i].neurons + j;

            for (int k = 0; k < inputsPerNeuron; ++k)
                neuron->w[k] = randomDouble(-1, 1);
        }
    }
}

void MultilayerPerceptron::applySigmoid(int layerIndex)
{
    for (int j = 0; j < layers[layerIndex].nOfNeurons; ++j) {
        Neuron* neuron = layers[layerIndex].neurons + j;
        double net = neuron->w[0];

        for (int k = 0; k < layers[layerIndex - 1].nOfNeurons; ++k)
            net += neuron->w[k + 1] * layers[layerIndex - 1].neurons[k].out; //We add 1 to the weight index to account for the bias w_0

        neuron->out = sigmoid(net);
    }
}

void MultilayerPerceptron::applySoftmax(int layerIndex)
{
    double *net = new double[layers[layerIndex].nOfNeurons], sumatoryOfExp = 0;

    for (int j = 0; j < layers[layerIndex].nOfNeurons; ++j) {
        Neuron* neuron = layers[layerIndex].neurons + j;
        net[j] = neuron->w[0];

        for (int k = 0; k < layers[layerIndex - 1].nOfNeurons; ++k)
            net[j] += neuron->w[k + 1] * layers[layerIndex - 1].neurons[k].out; //We add 1 to the weight index to account for the bias w_0

        sumatoryOfExp += exp(net[j]);
    }

    for (int j = 0; j < layers[layerIndex].nOfNeurons; ++j)
        layers[layerIndex].neurons[j].out = exp(net[j]) / sumatoryOfExp;

    delete[] net;
}

double MultilayerPerceptron::getMSE(const double* target)
{
    double* prediction = new double[layers[nOfLayers - 1].nOfNeurons];

    getOutputs(prediction);

    double squaredSum = .0;

    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i)
        squaredSum += pow((prediction[i] - target[i]), 2);

    delete[] prediction;
    return squaredSum / layers[nOfLayers - 1].nOfNeurons;
}

double MultilayerPerceptron::getCE(const double* target)
{
    double* prediction = new double[layers[nOfLayers - 1].nOfNeurons];
    getOutputs(prediction);

    double sumatory = .0;

    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i)
        sumatory -= (target[i] * log(prediction[i]));

    delete[] prediction;

    return sumatory / layers[nOfLayers - 1].nOfNeurons;
}

void MultilayerPerceptron::feedInputs(double* input)
{
    for (int i = 0; i < layers[0].nOfNeurons; ++i)
        layers[0].neurons[i].out = input[i];
}

void MultilayerPerceptron::getOutputs(double* output)
{
    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i)
        output[i] = layers[nOfLayers - 1].neurons[i].out;
}

void MultilayerPerceptron::copyWeights()
{
    for (int layerIndex = 1; layerIndex < nOfLayers; ++layerIndex) {
        Layer* layer = layers + layerIndex;
        for (int neuronIndex = 0; neuronIndex < layer->nOfNeurons; ++neuronIndex) {
            Neuron* neuron = layer->neurons + neuronIndex;
            int nWeights = layers[layerIndex - 1].nOfNeurons + 1;

            memcpy(neuron->wCopy, neuron->w, sizeof(double) * nWeights);
        }
    }
}

void MultilayerPerceptron::restoreWeights()
{
    for (int layerIndex = 1; layerIndex < nOfLayers; ++layerIndex) {
        Layer* layer = layers + layerIndex;
        for (int neuronIndex = 0; neuronIndex < layer->nOfNeurons; ++neuronIndex) {
            Neuron* neuron = layer->neurons + neuronIndex;
            int nWeights = layers[layerIndex - 1].nOfNeurons + 1;

            memcpy(neuron->w, neuron->wCopy, sizeof(double) * nWeights);
        }
    }
}

void MultilayerPerceptron::forwardPropagate()
{
    for (int i = 1; i < nOfLayers - 1; ++i)
        applySigmoid(i);

    switch (outputFunction) {
    case 0:
        applySigmoid(nOfLayers - 1);
        break;
    case 1:
        applySoftmax(nOfLayers - 1);
        break;
    default:
        std::cerr << "Invalid output function" << std::endl;
        exit(EXIT_FAILURE);
    }
}

double MultilayerPerceptron::obtainError(double* target, int errorFunction)
{
    switch (errorFunction) {
    case 0:
        return getMSE(target);
    case 1:
        return getCE(target);
    default:
        std::cerr << "Código de función de error inválido. Saliendo." << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MultilayerPerceptron::backpropagateError(double* target, int errorFunction)
{
    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i) {
        Neuron* neuron = layers[nOfLayers - 1].neurons + i;
        if (outputFunction == 0) { // For sigmoid output
            if (errorFunction == 0)
                neuron->delta = -(target[i] - neuron->out) * neuron->out * (1 - neuron->out);
            else
                neuron->delta = -(target[i] / neuron->out) * neuron->out * (1 - neuron->out);
        } else { //For softmax output
            neuron->delta = 0;
            for (int j = 0; j < layers[nOfLayers - 1].nOfNeurons; ++j) {
                Neuron* otherNeuron = layers[nOfLayers - 1].neurons + j;
                int aux = (i == j) ? 1 : 0;
                if (errorFunction == 0)
                    neuron->delta -= (target[j] - otherNeuron->out) * neuron->out * (aux - otherNeuron->out);
                else
                    neuron->delta -= (target[j] / otherNeuron->out) * neuron->out * (aux - otherNeuron->out);
            }
        }

        for (int h = nOfLayers - 2; h > 0; --h) {
            for (int i = 0; i < layers[h].nOfNeurons; ++i) {
                Neuron* neuron = layers[h].neurons + i;
                double sumatory = 0;
                for (int j = 0; j < layers[h + 1].nOfNeurons; ++j) {
                    Neuron* otherNeuron = layers[h + 1].neurons + j;
                    sumatory += otherNeuron->delta * otherNeuron->w[i + 1];
                }
                neuron->delta = sumatory * neuron->out * (1 - neuron->out);
            }
        }
    }
}

void MultilayerPerceptron::accumulateChange()
{
    for (int i = 1; i < nOfLayers; ++i) {
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            Neuron* neuron = layers[i].neurons + j;
            for (int k = 0; k < layers[i - 1].nOfNeurons; ++k) {
                Neuron* prevLayerNeuron = layers[i - 1].neurons + k;
                neuron->deltaW[k + 1] += neuron->delta * prevLayerNeuron->out;
            }
            neuron->deltaW[0] += neuron->delta;
        }
    }
}

void MultilayerPerceptron::weightAdjustment(const int nOfPatterns)
{
    for (int i = 1; i < nOfLayers; ++i) {
        double decrement = pow(decrementFactor, -(nOfLayers - i + 1));
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            Neuron* neuron = layers[i].neurons + j;
            for (int k = 0; k < layers[i - 1].nOfNeurons; ++k)
                neuron->w[k + 1] -= (decrement * eta * neuron->deltaW[k + 1] / nOfPatterns) + (decrement * mu * eta * neuron->lastDeltaW[k + 1] / nOfPatterns);

            neuron->w[0] -= (decrement * eta * neuron->deltaW[0] / nOfPatterns) + (decrement * mu * eta * neuron->lastDeltaW[0] / nOfPatterns);
        }
    }
}

void MultilayerPerceptron::printNetwork()
{
    for (int i = 1; i < nOfLayers; i++) {
        std::cout << "Layer " << i << std::endl;
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
                if (!(outputFunction == 1 && (i == (nOfLayers - 1)) && (k == (layers[i].nOfNeurons - 1))))
                    std::cout << layers[i].neurons[j].w[k] << " ";
            std::cout << "\n";
        }
        std::cout << "\n\n";
    }
}

void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction)
{
    if (online)
        this->clearDeltaWeights();

    this->feedInputs(input);
    this->forwardPropagate();
    this->backpropagateError(target, errorFunction);
    this->accumulateChange();

    if (online)
        this->weightAdjustment();
    //printNetwork();
}

void MultilayerPerceptron::clearDeltaWeights()
{
    for (int i = 1; i < nOfLayers; ++i)
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            int nWeights = layers[i - 1].nOfNeurons + 1;
            for (int k = 0; k < nWeights; ++k) {
                memcpy(layers[i].neurons[j].lastDeltaW, layers[i].neurons[j].w, nWeights * sizeof(double));
                layers[i].neurons[j].deltaW[k] = 0;
            }
        }
}

Dataset* MultilayerPerceptron::readData(const char* fileName)
{
    Dataset* dataset = new Dataset;
    std::ifstream dataFile(fileName, std::ifstream::in);
    if (!dataFile.is_open()) {
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

void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction)
{

    if (!online)
        this->clearDeltaWeights();

    for (uint i = 0; i < nOfTrainingPatterns; ++i)
        performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);

    if (!online)
        this->weightAdjustment(nOfTrainingPatterns);
}

double MultilayerPerceptron::test(Dataset* dataset, int errorFunction)
{
    double accumulatedError = .0;

    for (int i = 0; i < dataset->nOfPatterns; ++i) {
        feedInputs(dataset->inputs[i]);
        forwardPropagate();

        accumulatedError += obtainError(dataset->outputs[i], errorFunction);
    }

    return (accumulatedError / dataset->nOfPatterns);
}

double MultilayerPerceptron::testClassification(Dataset* dataset)
{
    double* predictedOutputs = new double[layers[nOfLayers - 1].nOfNeurons];
    int correctGuesses = 0;
    for (int patternIndex = 0; patternIndex < dataset->nOfPatterns; ++patternIndex) {
        feedInputs(dataset->inputs[patternIndex]);
        forwardPropagate();
        getOutputs(predictedOutputs);

        int predictedClass = getMaxOutput(), correctClass = 0;

        for (int i = 0; i < dataset->nOfOutputs; ++i)
            if (dataset->outputs[patternIndex][i] > dataset->outputs[patternIndex][correctClass])
                correctClass = i;

        if (correctClass == predictedClass)
            correctGuesses += 1;
    }

    return (double)correctGuesses * 100.0 / dataset->nOfPatterns;
}

Dataset* MultilayerPerceptron::datasetFromIndexes(Dataset* dataset, int* indexes, int size)
{
    Dataset* result = new Dataset();

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

void MultilayerPerceptron::splitDataset(Dataset* dataset, Dataset** train, Dataset** validation)
{
    int valSize = validationRatio * nOfTrainingPatterns;
    int trainSize = nOfTrainingPatterns - valSize;
    int* trainingIndexes;
    int* validationIndexes = util::integerRandomVectorWithoutRepeating(0, dataset->nOfPatterns - 1, valSize, &trainingIndexes);

    *validation = datasetFromIndexes(dataset, validationIndexes, valSize);
    *train = datasetFromIndexes(dataset, trainingIndexes, trainSize);
}

void MultilayerPerceptron::predict(Dataset* dataset)
{
    int i;
    int numSalidas = layers[nOfLayers - 1].nOfNeurons;
    double* salidas = new double[numSalidas];

    cout << "Id,Category" << endl;

    for (i = 0; i < dataset->nOfPatterns; i++) {

        feedInputs(dataset->inputs[i]);
        forwardPropagate();
        getOutputs(salidas);

        int maxIndex = getMaxOutput();

        std::cout << i << "," << maxIndex << endl;
    }
}

void MultilayerPerceptron::runBackPropagation(Dataset* trainDataset, Dataset* testDataset, int maxiter, double* errorTrain, double* errorTest, double* ccrTrain, double* ccrTest, int errorFunction)
{
    int countTrain = 0;

    randomWeights();

    double minTrainError = 0;
    int iterWithoutImproving = 0;
    nOfTrainingPatterns = trainDataset->nOfPatterns;

    Dataset *validationDataset = NULL, *learningDataset = NULL;
    double validationError = 0, previousValidationError = 0;
    int iterWithoutImprovingValidation = 0;

    if (validationRatio > 0 && validationRatio < 1)
        splitDataset(trainDataset, &learningDataset, &validationDataset);
    else
        learningDataset = trainDataset;

    nOfTrainingPatterns = learningDataset->nOfPatterns;

    do {

        train(learningDataset, errorFunction);

        double trainError = test(learningDataset, errorFunction);
        if (countTrain == 0 || trainError < minTrainError) {
            minTrainError = trainError;
            copyWeights();
            iterWithoutImproving = 0;
        } else if ((trainError - minTrainError) < 0.00001)
            iterWithoutImproving = 0;
        else
            iterWithoutImproving++;

        if (iterWithoutImproving == 50) {
            cout << "We exit because the training is not improving!!" << endl;
            restoreWeights();
            countTrain = maxiter - 1;
        }

        countTrain++;

        if (validationDataset != NULL) {
            if (previousValidationError == 0)
                previousValidationError = 999999999.9999999999;
            else
                previousValidationError = validationError;
            validationError = test(validationDataset, errorFunction);
            if (validationError < previousValidationError)
                iterWithoutImprovingValidation = 0;
            else if ((validationError - previousValidationError) < 0.00001)
                iterWithoutImprovingValidation = 0;
            else
                iterWithoutImprovingValidation++;
            if (iterWithoutImprovingValidation == 50) {
                cout << "We exit because validation is not improving!!" << endl;
                restoreWeights();
                countTrain = maxiter - 1;
            }
        }
        cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

    } while (countTrain < maxiter);

    if ((iterWithoutImprovingValidation != 50) && (iterWithoutImproving != 50))
        restoreWeights();

    cout << "NETWORK WEIGHTS" << endl;
    cout << "===============" << endl;
    printNetwork();

    cout << "Desired output Vs Obtained output (test)" << endl;
    cout << "=========================================" << endl;
    for (int i = 0; i < testDataset->nOfPatterns; i++) {
        double* prediction = new double[testDataset->nOfOutputs];

        // Feed the inputs and propagate the values
        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(prediction);
        for (int j = 0; j < testDataset->nOfOutputs; j++)
            cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
        cout << endl;
        delete[] prediction;
    }

    *errorTest = test(testDataset, errorFunction);
    *errorTrain = minTrainError;
    *ccrTest = testClassification(testDataset);
    *ccrTrain = testClassification(learningDataset);
}

bool MultilayerPerceptron::saveWeights(const char* fileName)
{
    // Object for writing the file
    ofstream f(fileName);

    if (!f.is_open())
        return false;

    // Write the number of layers and the number of layers in every layer
    f << nOfLayers;

    for (int i = 0; i < nOfLayers; i++)
        f << " " << layers[i].nOfNeurons;

    f << " " << outputFunction;
    f << endl;

    // Write the weight matrix of every layer
    for (int i = 1; i < nOfLayers; i++)
        for (int j = 0; j < layers[i].nOfNeurons; j++)
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
                if (layers[i].neurons[j].w != NULL)
                    f << layers[i].neurons[j].w[k] << " ";

    f.close();

    return true;
}

bool MultilayerPerceptron::readWeights(const char* fileName)
{
    ifstream f(fileName);

    if (!f.is_open())
        return false;

    int nl;
    int* npl;

    f >> nl;

    npl = new int[nl];

    for (int i = 0; i < nl; i++) {
        f >> npl[i];
    }
    f >> outputFunction;

    initialize(nl, npl);

    for (int i = 1; i < nOfLayers; i++)
        for (int j = 0; j < layers[i].nOfNeurons; j++)
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
                if (!(outputFunction == 1 && (i == (nOfLayers - 1)) && (k == (layers[i].nOfNeurons - 1))))
                    f >> layers[i].neurons[j].w[k];

    f.close();
    delete[] npl;

    return true;
}
