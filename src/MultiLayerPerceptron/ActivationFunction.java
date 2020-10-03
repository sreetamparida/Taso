package MultiLayerPerceptron;

public interface ActivationFunction {

    public double evalute(double value);

    public double evaluteDerivate(double value);

    public double initiateWeight(double input, double output);
}
