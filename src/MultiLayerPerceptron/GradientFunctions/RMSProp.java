package MultiLayerPerceptron.GradientFunctions;
import MultiLayerPerceptron.ActivationFunction;
import MultiLayerPerceptron.GradientFunction;
import MultiLayerPerceptron.Neuron;

public class RMSProp implements GradientFunction {

    double learningRate = 0.001;
    double epsilon = 1e-6f;

    @Override
    public double evalute(double pastWeight, double diff, Neuron neuron) {
        neuron.pastSquareGrad = beta*neuron.pastSquareGrad + (1-beta)*Math.pow(diff,2);
        return pastWeight - learningRate/Math.sqrt(neuron.pastSquareGrad+epsilon)*diff;

    }
}
