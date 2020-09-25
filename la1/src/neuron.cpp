#include "neuron.hpp"
#include <cmath>
#include <vector>

using std::vector;

double Neuron::_weightedSum(std::vector<double> inputs)
{
    double sum = .0;
    for (size_t i = 0; i < inputs.size(); ++i)
        sum += inputs[i] * weight(i);
    return sum;
}

double Neuron::_sigmoid(double x) const
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
    _net = _weightedSum(inputs);
    _out = _sigmoid(_net);
}

void Neuron::backpropagate(double target)
{
    //delta_j^H = - (d_j - out__j^H) * g'(net_j^H)
    double derivative = _sigmoid(_net) * (1 - _sigmoid(_net));
    double error = target - _out;
    delta(-error * derivative);
}

void Neuron::backpropagate(const Layer& nextLayer, int ownIndex)
{
    //delta_j^H = - (Sum_i=0^n_h+1) * g'(net_j^H)
    double derivative = _sigmoid(_net) * (1 - _sigmoid(_net));
    double sumatory = .0;

    for (Neuron neuron : nextLayer.neurons())
        sumatory += neuron.delta() * neuron.weight(ownIndex);

    delta(sumatory * derivative);
}