package MultiLayerPerceptron;

public class Layer {
    Neuron[] neurons;
    ActivationFunction activationFunction;
    GradientFunction gradientFunction;
    int no_neurons;
    Layer nextLayer;
    private Neuron bias;

    Layer(int no_neurons, ActivationFunction activationFunction, GradientFunction gradientFunction){
        this.no_neurons = no_neurons;
        this.activationFunction = activationFunction;
        this.gradientFunction = gradientFunction;
        neurons = new Neuron[this.no_neurons];
    }

    Layer(int no_neurons,GradientFunction gradientFunction){
        this.no_neurons = no_neurons;
        this.gradientFunction = gradientFunction;
        neurons = new Neuron[this.no_neurons];
    }

    public void generateNeurons(Layer nextLayer, double weightInitiator){
        generateBias(nextLayer,1, weightInitiator);
        this.nextLayer = nextLayer;
        for (int i = 0; i < no_neurons; i++) {
            neurons[i] = new Neuron(activationFunction,gradientFunction);
            neurons[i].setNextLayer(nextLayer);
            neurons[i].setWeights(weightInitiator);
        }
    }

    public void generateBias(Layer nextLayer,double y_in, double weightInitiator){
        this.bias = new Neuron(gradientFunction);
        this.bias.y_out = 1;
        this.bias.setNextLayer(nextLayer);
        this.bias.setWeights(weightInitiator);
    }


    public void generateOutputNeuron(){
        for (int i = 0; i < no_neurons; i++) {
            neurons[i] = new Neuron(activationFunction,gradientFunction);
        }
    }



    public void feedForward(){
        int nextLayerCount = this.nextLayer.no_neurons;
        for (int i = 0; i < nextLayerCount ; i++) {
            double value = 0.0;
            for (int j = 0; j < no_neurons ; j++) {
                value += neurons[j].y_out*neurons[j].weights[i];
            }
            value += bias.weights[i]*bias.y_out;
            this.nextLayer.neurons[i].feedForward(value);
        }
    }

    public void backPropagate() {
        bias.updateBias();
        for (int i = 0; i < no_neurons ; i++) {
            neurons[i].backPropagate();
        }
    }

    public void backPropagate(int value){
        bias.updateBias();
        for (int i = 0; i < no_neurons ; i++) {
            neurons[i].updateBias();
        }
    }



}
