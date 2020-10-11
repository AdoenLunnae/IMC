#include "neuron.hpp"
#include <cmath>
#include <vector>

using std::vector;
using namespace imc;

Neuron::Neuron(const int& nOfInputs)
{

    _weights = *new vector<double>(nOfInputs);
    _weightsCopy = *new vector<double>(nOfInputs);
    _delta = 0;
    _deltaW = *new vector<double>(nOfInputs, 0);
    _lastDeltaW = *new vector<double>(nOfInputs, 0);
}

double Neuron::_weightedSum(std::vector<double> inputs)
{
    double sum = -_bias;
    for (unsigned int i = 0; i < inputs.size(); ++i)
        sum -= inputs[i] * weight(i);
    return sum;
}

double Neuron::_sigmoid(double x) const
{
    return 1 / (1 + exp(x));
}

void Neuron::clearDeltas()
{
    _deltaBias = .0;
    for (unsigned int i = 0; i < _deltaW.size(); ++i)
        _deltaW[i] = .0;
}

void Neuron::saveDeltas()
{
    _lastDeltaBias = _deltaBias;
    _lastDeltaW = _deltaW;
}

void Neuron::randomWeights(unsigned int numInputs)
{
    double randWeight;

    _weights = *new vector<double>(numInputs);
    _weightsCopy = *new vector<double>(numInputs);

    _bias = ((double)rand() / RAND_MAX) * 2 - 1;
    for (unsigned int i = 0; i < _weights.size(); ++i) {
        randWeight = ((double)rand() / RAND_MAX) * 2 - 1;
        weight(i, randWeight);
    }
}

void Neuron::copyWeights()
{
    _weightsCopy = _weights;
    _biasCopy = _bias;
}

void Neuron::restoreWeights()
{
    _weights = _weightsCopy;
    _bias = _biasCopy;
}

void Neuron::feed(std::vector<double> inputs)
{
    _net = _weightedSum(inputs);
    _out = _sigmoid(_net);
}

void Neuron::backpropagate(double target)
{
    double derivative = _sigmoid(_net) * (1 - _sigmoid(_net));
    double error = target - _out;
    delta(-error * derivative);
}

void Neuron::backpropagate(const Layer& nextLayer, int ownIndex)
{
    double derivative = _sigmoid(_net) * (1 - _sigmoid(_net));
    double sumatory = .0;

    for (const Neuron& neuron : nextLayer.neurons())
        sumatory += neuron.delta() * neuron.weight(ownIndex);

    delta(sumatory * derivative);
}

void Neuron::acummulateChange(const vector<double>& layerInput)
{
    for (unsigned int i = 0; i < _deltaW.size(); ++i)
        _deltaW[i] += delta() * layerInput[i];

    _deltaBias += delta();
}

void Neuron::weightAdjustement(const double& learningRate, const double& momentumRate)
{
    for (unsigned int i = 0; i < _weights.size(); ++i)
        _weights[i] = _weights[i] - learningRate * _deltaW[i] - momentumRate * learningRate * _lastDeltaW[i];

    _bias = _bias - learningRate * _deltaBias - momentumRate * learningRate * _lastDeltaBias;
}