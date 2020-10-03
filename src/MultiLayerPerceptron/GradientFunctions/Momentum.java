package MultiLayerPerceptron.GradientFunctions;
import MultiLayerPerceptron.GradientFunction;
import MultiLayerPerceptron.Neuron;

public class Momentum implements GradientFunction {
    @Override
    public double evalute(double pastWeight, double diff, Neuron neuron) {
        neuron.movingAverage = beta*neuron.movingAverage + (1-beta)*diff;
        return (pastWeight - learningRate*neuron.movingAverage);
    }
}
