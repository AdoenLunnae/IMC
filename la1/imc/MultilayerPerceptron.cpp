/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <cstring>
#include <limits>
#include <math.h>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	eta = .1;
	decrementFactor = 1.0;
	mu = .9;
	validationRatio = .0;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[])
{
	nOfLayers = nl;
	layers = new Layer[nl];

	for (int i = 0; i < nl; ++i)
	{

		layers[i] = *new Layer;
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[npl[i]];

		int inputsPerNeuron = i != 0 ? npl[i - 1] + 1 : 1; // +1 to account for the bias input

		//We initialize layer 0 weights too to avoid problems freeing memory
		for (int j = 0; j < layers[i].nOfNeurons; ++j)
		{
			Neuron *neuron = layers[i].neurons + j;

			neuron->w = new double[inputsPerNeuron];
			neuron->deltaW = new double[inputsPerNeuron];
			neuron->wCopy = new double[inputsPerNeuron];
			neuron->lastDeltaW = new double[inputsPerNeuron];

			for (int k = 0; k < inputsPerNeuron; ++k)
			{
				neuron->deltaW[k] = 0;
				neuron->lastDeltaW[k] = 0;
			}
		}
	}
	return 0;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
	freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
	delete[] layers;
}

Dataset *MultilayerPerceptron::datasetFromIndexes(Dataset *original, int *indexes, int size)
{
	Dataset *result = new Dataset;

	result->nOfPatterns = size;
	result->nOfInputs = original->nOfInputs;
	result->nOfOutputs = original->nOfOutputs;
	result->inputs = new double *[size];
	result->outputs = new double *[size];
	for (int i = 0; i < size; ++i)
	{
		result->inputs[i] = new double[result->nOfInputs];
		result->outputs[i] = new double[result->nOfOutputs];

		memcpy(result->inputs[i], original->inputs[indexes[i]], result->nOfInputs * sizeof(double));
		memcpy(result->outputs[i], original->outputs[indexes[i]], result->nOfOutputs * sizeof(double));
	}
	return result;
}

void MultilayerPerceptron::generateValidationData(Dataset *original, Dataset **tD, Dataset **vD)
{
	int nPatterns = original->nOfPatterns;

	int valSize = validationRatio * nPatterns;
	int trainSize = nPatterns - valSize;

	int *valIndexes, *trainIndexes = new int[trainSize];

	valIndexes = intRandomVectorWithoutRepeating(nPatterns - 1, valSize, trainIndexes);

	*tD = datasetFromIndexes(original, trainIndexes, trainSize);
	*vD = datasetFromIndexes(original, valIndexes, valSize);

	delete[] valIndexes;
	delete[] trainIndexes;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	int inputsPerNeuron = 1;

	for (int i = 1; i < nOfLayers; ++i)
	{
		inputsPerNeuron = layers[i - 1].nOfNeurons + 1; //To account for the bias
		for (int j = 0; j < layers[i].nOfNeurons; ++j)
		{
			Neuron *neuron = layers[i].neurons + j;

			for (int k = 0; k < inputsPerNeuron; ++k)
				neuron->w[k] = randomDouble(-1, 1);
		}
	}
}

void MultilayerPerceptron::clearDeltaWeights()
{
	for (int i = 1; i < nOfLayers; ++i)
		for (int j = 0; j < layers[i].nOfNeurons; ++j)
		{
			int nWeights = layers[i - 1].nOfNeurons + 1;
			memcpy(layers[i].neurons[j].lastDeltaW, layers[i].neurons[j].deltaW, nWeights * sizeof(double));
			for (int k = 0; k < nWeights; ++k)
				layers[i].neurons[j].deltaW[k] = 0;
		}
}

double MultilayerPerceptron::randomDouble(const double &min, const double &max)
{
	return (double)rand() / RAND_MAX * (max - min) + min;
}

Dataset *MultilayerPerceptron::permutateDataset(Dataset *dataset)
{
	int size = dataset->nOfPatterns;
	int *idx = intRandomVectorWithoutRepeating(size, size);
	Dataset *permutated = datasetFromIndexes(dataset, idx, size);

	return permutated;
}
// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input)
{
	for (int i = 0; i < layers[0].nOfNeurons; ++i)
		layers[0].neurons[i].out = input[i];
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output)
{
	for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i)
		output[i] = layers[nOfLayers - 1].neurons[i].out;
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	for (int layerIndex = 1; layerIndex < nOfLayers; ++layerIndex)
	{
		Layer *layer = layers + layerIndex;
		for (int neuronIndex = 0; neuronIndex < layer->nOfNeurons; ++neuronIndex)
		{
			Neuron *neuron = layer->neurons + neuronIndex;
			int nWeights = layers[layerIndex - 1].nOfNeurons + 1;

			memcpy(neuron->wCopy, neuron->w, sizeof(double) * nWeights);
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (int layerIndex = 1; layerIndex < nOfLayers; ++layerIndex)
	{
		Layer *layer = layers + layerIndex;
		int nWeights = layers[layerIndex - 1].nOfNeurons + 1;
		for (int neuronIndex = 0; neuronIndex < layer->nOfNeurons; ++neuronIndex)
		{
			Neuron *neuron = layer->neurons + neuronIndex;
			memcpy(neuron->w, neuron->wCopy, sizeof(double) * nWeights);
		}
	}
}
// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	for (int layerIndex = 1; layerIndex < nOfLayers; ++layerIndex)
	{
		Layer *layer = layers + layerIndex;
		Layer *prevLayer = layers + layerIndex - 1;
		for (int neuronIndex = 0; neuronIndex < layer->nOfNeurons; ++neuronIndex)
		{
			Neuron *neuron = layer->neurons + neuronIndex;
			double net = neuron->w[0];

			for (int k = 0; k < prevLayer->nOfNeurons; ++k)
				net += neuron->w[k + 1] * prevLayer->neurons[k].out; //We add 1 to the weight index to account for the bias w_0

			neuron->out = 1 / (1 + exp(-net));
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double *target)
{
	int size = layers[nOfLayers - 1].nOfNeurons;

	double *prediction = new double[size];

	getOutputs(prediction);

	double squaredSum = .0;

	for (int i = 0; i < size; ++i)
		squaredSum += pow((prediction[i] - target[i]), 2);

	delete[] prediction;

	return squaredSum / size;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double *target)
{
	//Last Layer, backpropagate target
	for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i)
	{
		Neuron *neuron = layers[nOfLayers - 1].neurons + i;
		neuron->delta = -(target[i] - neuron->out) * neuron->out * (1 - neuron->out);
	}

	//Rest of layers, backpropagate error
	for (int h = nOfLayers - 2; h > 0; --h)
	{
		for (int j = 0; j < layers[h].nOfNeurons; ++j)
		{
			Neuron *neuron = layers[h].neurons + j;
			double summation = 0;
			for (int i = 0; i < layers[h + 1].nOfNeurons; ++i)
			{
				Neuron *otherNeuron = layers[h + 1].neurons + i;
				summation += otherNeuron->delta * otherNeuron->w[j + 1];
			}
			neuron->delta = summation * neuron->out * (1 - neuron->out);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	for (int i = 1; i < nOfLayers; ++i)
	{
		for (int j = 0; j < layers[i].nOfNeurons; ++j)
		{
			Neuron *neuron = layers[i].neurons + j;
			for (int k = 0; k < layers[i - 1].nOfNeurons; ++k)
			{
				Neuron *prevLayerNeuron = layers[i - 1].neurons + k;
				neuron->deltaW[k + 1] += neuron->delta * prevLayerNeuron->out;
			}
			neuron->deltaW[0] += neuron->delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	for (int i = 1; i < nOfLayers; ++i)
	{
		Layer *layer = layers + i, *prevLayer = layers + i - 1;

		double decrement = pow(decrementFactor, -(nOfLayers - i + 1));
		for (int j = 0; j < layer->nOfNeurons; ++j)
		{
			Neuron *neuron = layer->neurons + j;

			for (int k = 0; k < prevLayer->nOfNeurons; ++k)
				neuron->w[k + 1] -= ((decrement * eta * neuron->deltaW[k + 1]) + (decrement * mu * eta * neuron->lastDeltaW[k + 1]));

			neuron->w[0] -= (decrement * eta * neuron->deltaW[0]) + (decrement * mu * eta * neuron->lastDeltaW[0]);
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
	for (int i = 1; i < nOfLayers; i++)
	{
		std::cout << "Layer " << i << std::endl;
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				//if (!((i == (nOfLayers - 1)) && (k == (layers[i].nOfNeurons - 1))))
				std::cout << layers[i].neurons[j].w[k] << " ";
			std::cout << "\n";
		}
		std::cout << "\n\n";
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double *input, double *target)
{
	feedInputs(input);
	forwardPropagate();
	backpropagateError(target);
	accumulateChange();
	weightAdjustment();
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *MultilayerPerceptron::readData(const char *fileName)
{
	Dataset *dataset = new Dataset;
	std::ifstream dataFile(fileName, std::ifstream::in);
	if (!dataFile.is_open())
	{
		std::cerr << "Error opening file " << fileName << ". Exiting.\n";
		exit(EXIT_FAILURE);
	}

	dataFile >> dataset->nOfInputs;
	dataFile >> dataset->nOfOutputs;
	dataFile >> dataset->nOfPatterns;
	dataset->inputs = new double *[dataset->nOfPatterns];
	dataset->outputs = new double *[dataset->nOfPatterns];

	for (uint patternIndex = 0; patternIndex < dataset->nOfPatterns; ++patternIndex)
	{
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

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset *trainDataset)
{
	double **inputs = trainDataset->inputs;
	double **outputs = trainDataset->outputs;

	for (int i = 0; i < trainDataset->nOfPatterns; i++)
	{
		clearDeltaWeights();
		performEpochOnline(inputs[i], outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset *testDataset)
{
	double error = 0;

	double **inputs = testDataset->inputs;
	double **outputs = testDataset->outputs;

	for (int i = 0; i < testDataset->nOfPatterns; ++i)
	{
		feedInputs(inputs[i]);
		forwardPropagate();
		error += obtainError(outputs[i]);
	}
	return error / testDataset->nOfPatterns;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset *testDataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *obtained = new double[numSalidas];

	cout << "Id,Predicted" << endl;

	for (i = 0; i < testDataset->nOfPatterns; i++)
	{

		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);

		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;
	}
	delete[] obtained;
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset *trainDataset, Dataset *testDataset, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)

	double minTrainError = 0, previousValError = 0;
	int iterWithoutImprovingTrain, iterWithoutImprovingVal;
	double testError = 0, trainError;

	double valError = 0;

	Dataset *t, *v = NULL;
	// Generate validation data
	if (validationRatio > 0 && validationRatio < 1)
	{
		generateValidationData(trainDataset, &t, &v);
	}
	else
		t = trainDataset;

	// Learning
	randomWeights();
	do
	{
		trainOnline(t);

		trainError = test(t);

		if (countTrain == 0 || (minTrainError - trainError) > 1e-5)
		{
			minTrainError = trainError;
			copyWeights();
			iterWithoutImprovingTrain = 0;
		}
		else
			iterWithoutImprovingTrain++;

		if (iterWithoutImprovingTrain == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			countTrain = maxiter - 1;
		}

		if (v != NULL)
		{
			if (previousValError == 0)
				previousValError = 999999999.9999999999;
			else
				previousValError = valError;

			valError = test(v);

			if (previousValError - valError > 1e-5)
				iterWithoutImprovingVal = 0;
			else
				iterWithoutImprovingVal++;

			if (iterWithoutImprovingVal == 50)
			{
				cout << "We exit because validation is not improving!!" << endl;
				restoreWeights();
				countTrain = maxiter - 1;
			}
		}

		countTrain++;

		// Check validation stopping condition and force it
		// BE CAREFUL: in this case, we have to save the last validation error, not the minimum one
		// Apart from this, the way the stopping condition is checked is the same than that
		// applied for the training set
		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << valError << endl;

	} while (countTrain < maxiter);

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();
	/*
	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < testDataset->nOfPatterns; i++)
	{
		double *prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;
	}
	*/
	testError = test(testDataset);
	*errorTest = testError;
	*errorTrain = minTrainError;

	delete t;
	if (v != NULL)
		delete v;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
