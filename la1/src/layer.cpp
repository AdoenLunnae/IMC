#include "layer.hpp"
#include "neuron.hpp"
#include <vector>

using std::vector;
using namespace imc;

Layer::Layer(const int& nOfNeurons, const int& nOfInputs)
{
    _neurons = *new std::vector<Neuron>(nOfNeurons, *new Neuron(nOfInputs));
}

void Layer::clearDeltas()
{
    for (Neuron& neuron : _neurons)
        neuron.clearDeltas();
}

void Layer::saveDeltas()
{
    for (Neuron& neuron : _neurons)
        neuron.saveDeltas();
}

void Layer::feed(vector<double> inputs)
{
    _out.clear();
    _input = inputs;
    for (Neuron& neuron : _neurons) {
        neuron.feed(inputs);
        _out.push_back(neuron.out());
    }
}

void Layer::randomWeights(unsigned int numInputs)
{
    for (Neuron& neuron : _neurons)
        neuron.randomWeights(numInputs);
}

void Layer::copyWeights()
{
    for (Neuron& neuron : _neurons)
        neuron.copyWeights();
}

void Layer::restoreWeights()
{
    for (Neuron& neuron : _neurons)
        neuron.restoreWeights();
}

void Layer::backpropagate(const Layer& nextLayer)
{
    for (int i = 0; i < numberOfNeurons(); ++i)
        _neurons[i].backpropagate(nextLayer, i);
}

void Layer::backpropagate(double* target)
{
    for (int i = 0; i < numberOfNeurons(); ++i)
        _neurons[i].backpropagate(target[i]);
}

void Layer::accumulateChange()
{
    for (Neuron& neuron : _neurons)
        neuron.acummulateChange(_input);
}

void Layer::weightAdjustement(const double& learningRate, const double& momentumRate)
{
    for (Neuron& neuron : _neurons)
        neuron.weightAdjustement(learningRate, momentumRate);
}

vector<vector<double>> Layer::weightMatrix() const
{
    vector<vector<double>> matrix = *new vector<vector<double>>(numberOfNeurons());
}