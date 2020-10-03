package MultiLayerPerceptron.ActivationFunctions;
import MultiLayerPerceptron.ActivationFunction;

public class LeakyRelu implements ActivationFunction {
    @Override
    public double evalute(double value)
    {
        if(value >= 0.0)
            return 1.0;
        else
            return -0.1*value;
    }

    @Override
    public double evaluteDerivate(double value)
    {
        if(value >= 0.0)
            return 1.0;
        else
            return -0.1;
    }

    @Override
    public double initiateWeight(double input, double output) {
        double value = input + output;
        value = Math.sqrt(7/value);
        return value;
    }
}
