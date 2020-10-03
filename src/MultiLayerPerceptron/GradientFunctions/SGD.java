package MultiLayerPerceptron.GradientFunctions;

import MultiLayerPerceptron.GradientFunction;
import MultiLayerPerceptron.Neuron;

public class SGD implements GradientFunction {


    @Override
    public double evalute(double pastWeight, double diff, Neuron neuron) {
        return (pastWeight- learningRate*diff);
    }
}
