#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>

#ifndef LAYER_HPP
#include "layer.hpp"
#else
class Layer;
#endif
class Neuron {
private:
    std::vector<double> _weights;
    std::vector<double> _weightsCopy;

    std::vector<double> _deltaW;
    std::vector<double> _lastDeltaW;

    double _out;
    double _delta;
    double _net;

    double _weightedSum(std::vector<double> inputs);
    double _sigmoid(double x) const;

public:
    Neuron() {}
    ~Neuron() {}

    inline double weight(size_t index) const { return _weights[index]; }
    inline void weight(size_t index, double value) { _weights[index] = value; }

    inline double delta() const { return _delta; }
    inline void delta(double value) { _delta = value; }

    inline double out() const { return _out; }

    void randomWeights(size_t numInputs);

    void feed(std::vector<double> inputs);

    void backpropagate(double target);
    void backpropagate(const Layer& nextLayer, int ownIndex)
};
#endif /* NEURON_HPP */
