#ifndef LAYER_HPP
#define LAYER_HPP
#include <iostream>
#include <vector>

#ifndef NEURON_HPP
#include "neuron.hpp"
#else
namespace imc {
class Neuron;
}
#endif
namespace imc {
class Layer {
private:
    std::vector<Neuron> _neurons;
    std::vector<double> _out;
    std::vector<double> _input;

public:
    Layer() {};
    Layer(const int& nOfNeurons, const int& nOfInputs);
    unsigned int numberOfNeurons() const { return _neurons.size(); }
    inline std::vector<Neuron>& neurons() { return _neurons; }
    std::vector<double> out() const { return _out; };

    void clearDeltas();
    void saveDeltas();
    void randomWeights(unsigned int numInputs);

    void copyWeights();
    void restoreWeights();

    void feed(std::vector<double> inputs, bool isInput = false);

    void backpropagate(Layer& nextLayer);
    void backpropagate(double* target);

    void accumulateChange();
    void weightAdjustement(const double& learningRate, const double& momentumRate);

    void printMatrix(std::ostream& stream = std::cout) const;
};
};
#endif /* LAYER_HPP */
