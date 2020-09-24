#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>

class Neuron {
private:
    std::vector<double> _weights;
    std::vector<double> _weightsCopy;

    std::vector<double> _deltaW;
    std::vector<double> _lastDeltaW;

    double _out;
    double _delta;

    double _net(std::vector<double> inputs) const;
    double _sigmoid(double x) const;

public:
    Neuron() {}
    ~Neuron() {}

    inline double weight(size_t index) const { return _weights[index]; }
    inline void weight(size_t index, double value) { _weights[index] = value; }

    inline double out() const { return _out; }

    void randomWeights(size_t numInputs);

    void feed(std::vector<double> inputs);
};
#endif /* NEURON_HPP */
