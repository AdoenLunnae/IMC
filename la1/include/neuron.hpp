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
    Neuron() { }
    Neuron(const int& nOfInputs);
    ~Neuron() { }

    //Access and edit the weights
    inline double weight(unsigned int index) const { return _weights[index]; }
    inline void weight(unsigned int index, double value) { _weights[index] = value; }

    //Access and edit the delta
    inline double delta() const { return _delta; }
    inline void delta(double value) { _delta = value; }

    //Get the output vector
    inline double out() const { return _out; }

    //Initialize weights between -1 and 1
    void randomWeights(unsigned int numInputs);

    //Reset the deltaW and deltaBias
    void clearDeltas();

    //Save deltaW and deltaBias to lastDeltaW and lastDeltaBias
    void saveDeltas();

    //Save and restore the weights
    void copyWeights();
    void restoreWeights();

    //Calculate the output for some inputs
    void feed(std::vector<double> inputs);

    //Copy the input to the output
    void passInput(double input)
    {
        _out = input;
    }

    //Backpropagate error(output layer)
    void backpropagate(double target);

    //Backpropagate error(hidden layer)
    void backpropagate(Layer& nextLayer, int ownIndex);

    //Update deltaW
    void acummulateChange(const std::vector<double>& layerInput);

    //Update the weights
    void weightAdjustement(const double& learningRate, const double& momentumRate);

    //Get the weight vector
    std::vector<double> weights() const { return _weights; }

    void weights(const std::vector<double> val) { _weights = val; }

    //Get the bias
    double bias() const { return _bias; }
    void bias(const double& bias) { _bias = bias; }

    void readWeights(std::ifstream& file, const int& nInputs);
};
};
#endif /* NEURON_HPP */
