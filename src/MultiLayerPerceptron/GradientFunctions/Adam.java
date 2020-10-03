package MultiLayerPerceptron.GradientFunctions;
import MultiLayerPerceptron.GradientFunction;
import MultiLayerPerceptron.Neuron;

public class Adam implements GradientFunction {
    @Override
    public double evalute(double pastWeight, double diff, Neuron neuron) {
        double beta1 = 0.9;
        double beta2 = 0.999;
        double learningRate = 0.001;
        double epsilon = 1e-8f;
        double movAvg = 0.0;
        double pastGrad = 0.0;
        neuron.movingAverage = neuron.movingAverage*beta1 + (1-beta1)*diff;
        neuron.pastSquareGrad = neuron.pastSquareGrad*beta2 + (1-beta2)*Math.pow(diff,2);
        movAvg = neuron.movingAverage/(1-beta1);
        pastGrad = neuron.pastSquareGrad/(1-beta2);
        return pastWeight - (learningRate/(Math.sqrt(pastGrad)+epsilon))*movAvg;
    }
}
