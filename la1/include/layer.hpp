#ifndef LAYER_HPP
#define LAYER_HPP
#include "neuron.hpp"
#include <vector>

class Layer {
private:
    std::vector<Neuron> _neurons;

public:
    Layer(int nOfNeurons) { _neurons = *new std::vector<Neuron>(nOfNeurons); }

    void feed(std::vector<double> inputs);
};
#endif /* LAYER_HPP */
