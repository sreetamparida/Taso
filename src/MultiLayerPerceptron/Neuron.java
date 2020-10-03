package MultiLayerPerceptron;

import java.util.Random;

public class Neuron {
    private double err_prop;
    private Layer next_layer;
    double[] weights;
    private double[] newWeight;
    public double y_out;
    private ActivationFunction activationFunction;
    public GradientFunction gradientFunction;
    public double pastSquareGrad = 0;
    public double movingAverage = 0;

    Neuron(ActivationFunction activationFunction,GradientFunction gradientFunction) {
        this.activationFunction = activationFunction;
        this.gradientFunction = gradientFunction;
    }

    Neuron(GradientFunction gradientFunction){
        this.gradientFunction = gradientFunction;
    }


    public void setNextLayer(Layer layer) {
        this.next_layer = layer;
    }

    public void setWeights(double weightInitiator) {
        weights = new double[next_layer.no_neurons];
        newWeight = new double[next_layer.no_neurons];
        Random random = new Random();
        for(int i=0; i < next_layer.no_neurons; i++){
            weights[i] = (random.nextInt((int)((2*weightInitiator)*10+1))-weightInitiator*10)/10.0;
        }
    }


    public void updateWeights(){
        for(int i=0; i < next_layer.no_neurons; i++){
            double diff = next_layer.neurons[i].err_prop*y_out;
            double pastWeight = this.weights[i];
            this.newWeight[i] = gradientFunction.evalute(pastWeight,diff,this);
        }
    }

    public void setNewWeight(){
        int neuronCount = weights.length;
        for(int i=0;i<neuronCount;i++){
            weights[i] = newWeight[i];
        }
    }

    public void setErrProp() {
        this.err_prop = 0.0;
        for(int i=0; i < next_layer.no_neurons; i++) {
            this.err_prop += next_layer.neurons[i].err_prop*weights[i];
        }
        this.err_prop = err_prop*activationFunction.evaluteDerivate(y_out);
    }

    public double applyLoss(double target, double actual) {
        double adjustedTarget = (target == 0.0 ? 0.000001f : target);
        adjustedTarget = (target == 1.0 ? 0.999999f : adjustedTarget);
        double adjustedActual = (actual == 0 ? 0.000001f : actual);
        adjustedActual = (actual == 1 ? 0.999999f : adjustedActual);

        return -adjustedTarget / adjustedActual + (1 - adjustedTarget)
                / (1 - adjustedActual);
    }

    public double applyCrossEntropy(double target, double actual) {
        double epsilon = 1e-8f;
        return -target * (double) Math.log(Math.max(actual, epsilon)) - (1 - target)
                * (double) Math.log(Math.max(1 - actual, epsilon));
    }

    public int checkOutput(double target){
//        System.out.println("output = " + y_out + "   target = " + target);
        if(y_out == target){
            return 0;
        }
        this.err_prop = -2*Math.pow((target-y_out),1)*activationFunction.evaluteDerivate(y_out);
//        this.err_prop = applyLoss(target,y_out)*activationFunction.evaluteDerivate(y_out);
        return 1;
    }

    public void updateBias(){
        updateWeights();
        setNewWeight();
    }

    public void backPropagate() {
        updateWeights();
        setErrProp();
        setNewWeight();
    }

    public void feedForward(double y_in) {
        this.y_out = activationFunction.evalute(y_in);
    }

}
