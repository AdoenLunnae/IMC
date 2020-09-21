#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>

class Neuron {
private:
    std::vector<double> _weights;
    std::vector<double> _weightsCopy;

    std::vector<double> _deltaW;
    std::vector<double> _lastDeltaW;

    double _net(std::vector<double> inputs);
    double _sigmoid(double x);
    double _out;
    double _delta;

public:
    Neuron() {}
    ~Neuron() {}

    inline double weight(size_t index) { return _weights[index]; }
    inline void weight(size_t index, double value) { _weights[index] = value; }

    inline double out() { return _out; }

    void randomWeights(size_t numInputs);

    void feed(std::vector<double> inputs);
};
#endif /* NEURON_HPP */
