package MultiLayerPerceptron.ActivationFunctions;
import MultiLayerPerceptron.ActivationFunction;

public class Relu implements ActivationFunction {
    @Override
    public double evalute(double value)
    {
        if(value >= 0.0)
            return 1.0;
        else
            return 0.0;
    }

    @Override
    public double evaluteDerivate(double value)
    {
        return 1.0;
    }

    @Override
    public double initiateWeight(double input, double output) {
        double value = input + output;
        value = Math.sqrt(7/value);
        return value;
    }
}
