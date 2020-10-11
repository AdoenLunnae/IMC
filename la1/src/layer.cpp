#include "layer.hpp"
#include "neuron.hpp"
#include <iostream>
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

void Layer::feed(vector<double> inputs, bool isInput)
{
    _out.clear();
    _input = inputs;
    for (int i = 0; i < _neurons.size(); ++i) {
        if (!isInput)
            _neurons[i].feed(inputs);
        else
            _neurons[i].passInput(inputs[i]);

        _out.push_back(_neurons[i].out());
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

void Layer::backpropagate(Layer& nextLayer)
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

void Layer::printMatrix(std::ostream& stream) const
{
    for (const Neuron& neuron : _neurons) {
        stream << neuron.bias() << "\t";
        for (int i = 0; i < _input.size(); ++i)
            stream << neuron.weight(i) << "\t";
        stream << std::endl;
    }
}