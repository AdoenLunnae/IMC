#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>

#ifndef LAYER_HPP
#include "layer.hpp"
#else
namespace imc {
class Layer;
}
#endif
namespace imc {
class Neuron {
private:
    std::vector<double> _weights;
    std::vector<double> _weightsCopy;

    std::vector<double> _deltaW;
    std::vector<double> _lastDeltaW;

    double _bias;
    double _biasCopy;

    double _deltaBias;
    double _lastDeltaBias;

    double _out;
    double _delta;
    double _net;

    double _weightedSum(std::vector<double> inputs);
    double _sigmoid(double x) const;

public:
    Neuron() {}
    Neuron(const int& nOfInputs);
    ~Neuron() {}

    inline double weight(unsigned int index) const { return _weights[index]; }
    inline void weight(unsigned int index, double value) { _weights[index] = value; }

    inline double delta() const { return _delta; }
    inline void delta(double value) { _delta = value; }

    inline double out() const { return _out; }

    void randomWeights(unsigned int numInputs);

    void clearDeltas();
    void saveDeltas();

    void copyWeights();
    void restoreWeights();

    void feed(std::vector<double> inputs);

    void backpropagate(double target);
    void backpropagate(const Layer& nextLayer, int ownIndex);

    void acummulateChange(const std::vector<double>& layerInput);
    void weightAdjustement(const double& learningRate, const double& momentumRate);

    std::vector<double> weights() const { return _weights; }
};
};
#endif /* NEURON_HPP */
