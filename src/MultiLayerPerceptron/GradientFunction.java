package MultiLayerPerceptron;

public interface GradientFunction {
    public double movingAvg = 0;
    public double pastSquaredGrad = 0;
    public double beta = 0.9;
    public double epsilon = 1e-8f;
    public double learningRate = 0.1;
    public double evalute(double pastWeight, double diff, Neuron neuron);
}
