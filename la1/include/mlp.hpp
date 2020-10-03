#ifndef MLP_HPP
#define MLP_HPP

#include "layer.hpp"

namespace imc {
struct Dataset {
    int nOfInputs; /* Number of inputs */
    int nOfOutputs; /* Number of outputs */
    int nOfPatterns; /* Number of patterns */
    double** inputs; /* Matrix with the inputs of the problem */
    double** outputs; /* Matrix with the outputs of the problem */
};

class MLP {
private:
    std::vector<Layer> _layers;
    // Free memory for the data structures
    void _freeMemory();

    void _clearDeltas();

    // Feel all the weights (w) with random numbers between -1 and +1
    void _randomWeights();

    // Feed the input neurons of the network with a vector passed as an argument
    void _feedInputs(double* input);

    // Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
    void _getOutputs(double** output);

    // Make a copy of all the weights (copy w in wCopy)
    void _copyWeights();

    // Restore a copy of all the weights (copy wCopy in w)
    void _restoreWeights();

    // Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
    void _forwardPropagate();

    // Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
    double _obtainError(double* target);

    // Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
    void _backpropagateError(double* target);

    // Accumulate the changes produced by one pattern and save them in deltaW
    void _accumulateChange();

    // Update the network weights, from the first layer to the last one
    void _weightAdjustment();

    // Print the network, i.e. all the weight matrices
    void _printNetwork();

    // Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
    // input is the input vector of the pattern and target is the desired output vector of the pattern
    void _performEpochOnline(double* input, double* target);

    //Returns a read/write reference to the output layer
    Layer& _lastLayer() { return _layers[_layers.size() - 1]; }

    double* _outputPointer();

public:
    // Values of the parameters (they are public and can be updated from outside)
    double eta; // Learning rate
    double mu; // Momentum factor
    double validationRatio; // Ratio of training patterns used as validation
    double decrementFactor; // Decrement factor used for eta in the different layers

    // Constructor: Default values for all the parameters
    MLP();

    // DESTRUCTOR: free memory
    ~MLP();

    // Allocate memory for the data structures
    // nl is the number of layers and npl is a vetor containing the number of neurons in every layer
    // Give values to Layer* layers
    int initialize(int nl, int npl[]);

    // Read a dataset from a file name and return it
    Dataset* readData(const char* fileName);

    // Test the network with a dataset and return the MSE
    double test(Dataset* dataset);

    // Obtain the predicted outputs for a dataset
    void predict(Dataset* testDataset);

    // Perform an online training for a specific dataset
    void trainOnline(Dataset* trainDataset);

    // Run the traning algorithm for a given number of epochs, using trainDataset
    // Once finished, check the performance of the network in testDataset
    // Both training and test MSEs should be obtained and stored in errorTrain and errorTest
    void runOnlineBackPropagation(Dataset* trainDataset, Dataset* testDataset, int maxiter, double* errorTrain, double* errorTest);

    // Optional Kaggle: Save the model weights in a textfile
    bool saveWeights(const char* archivo);

    // Optional Kaggle: Load the model weights from a textfile
    bool readWeights(const char* archivo);
};
};
#endif