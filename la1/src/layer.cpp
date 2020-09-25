#include "layer.hpp"
#include "neuron.hpp"
#include <vector>
using std::vector;

void Layer::feed(vector<double> inputs)
{
    for (Neuron neuron : _neurons)
        neuron.feed(inputs);
}

void Layer::randomWeights(size_t numInputs)
{
    for (Neuron neuron : _neurons)
        neuron.randomWeights(numInputs);
}

vector<double> Layer::out() const
{
    vector<double> output;

    for (Neuron neuron : _neurons)
        output.push_back(neuron.out());

    return output;
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