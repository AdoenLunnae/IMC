#include "neuron.hpp"
#include <cmath>
#include <vector>

using std::vector;

double Neuron::_net(std::vector<double> inputs)
{
    double sum = .0;
    for (size_t i = 0; i < inputs.size(); ++i)
        sum += inputs[i] * weight(i);
    return sum;
}

double Neuron::_sigmoid(double x)
{
    return 1 / (1 + exp(x));
}

void Neuron::randomWeights(size_t numInputs)
{
    double randWeight;
    _weights = *new vector<double>(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        randWeight = ((double)rand() / RAND_MAX) * 2 - 1;
        weight(i, randWeight);
    }
}

void Neuron::feed(std::vector<double> inputs)
{
    _out = _sigmoid(_net(inputs));
}