#ifndef LAYER_HPP
#define LAYER_HPP
#include <vector>

#ifndef NEURON_HPP
#include "neuron.hpp"
#else
class Neuron;
#endif
class Layer {
private:
    std::vector<Neuron> _neurons;

public:
    Layer(int nOfNeurons) { _neurons = *new std::vector<Neuron>(nOfNeurons); }

    uint numberOfNeurons() const { return _neurons.size(); }
    inline const std::vector<Neuron>& neurons() const { return _neurons; }
    std::vector<double> out() const;

    void randomWeights(size_t numInputs);

    void feed(std::vector<double> inputs);

    void backpropagate(const Layer& nextLayer);
    void backpropagate(double* target);
};
#endif /* LAYER_HPP */
