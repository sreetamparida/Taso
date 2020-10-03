package MultiLayerPerceptron.ActivationFunctions;

import MultiLayerPerceptron.ActivationFunction;

public class Tanh implements ActivationFunction {

    @Override
    public double evalute(double value)
    {
        return Math.tanh(value);
    }

    @Override
    public double evaluteDerivate(double value)
    {
        return 1 - Math.pow(value, 2);
    }

    @Override
    public double initiateWeight(double input, double output) {
        double value = input + output;
        value = Math.sqrt(6/value);
        return value;
    }
}
