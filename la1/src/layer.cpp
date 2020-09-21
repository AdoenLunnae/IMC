#include "layer.hpp"
#include "neuron.hpp"
#include <vector>
using std::vector;

void Layer::feed(vector<double> inputs)
{
    for (Neuron neuron : _neurons)
        neuron.feed(inputs);
}