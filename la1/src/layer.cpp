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