package MultiLayerPerceptron.ActivationFunctions;

import MultiLayerPerceptron.ActivationFunction;


public class Sigmoid implements ActivationFunction {

    @Override
    public double evalute(double value) {
        return 1 / (1 + Math.pow(Math.E, - value));
    }

    @Override
    public double evaluteDerivate(double value) {
        return (value - Math.pow(value, 2));
    }

    @Override
    public double initiateWeight(double input, double output) {
        double value = input + output;
        value = 4*Math.sqrt(6/value);
        return value;
    }
}
