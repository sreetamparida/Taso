package MultiLayerPerceptron.GradientFunctions;
import MultiLayerPerceptron.GradientFunction;
import MultiLayerPerceptron.Neuron;

public class AdaGrad implements GradientFunction{
    double learningRate = 0.01;
    double epsilon = 1e-7f;
    @Override
    public double evalute(double pastWeight, double diff, Neuron neuron) {
        learningRate = 0.01;
//        System.out.println("-------------"+neuron.pastSquareGrad);
        neuron.pastSquareGrad = neuron.pastSquareGrad + Math.pow(diff,2);
        double update = pastWeight - (learningRate/Math.sqrt(neuron.pastSquareGrad+epsilon))*diff;

        return update;
    }
}
