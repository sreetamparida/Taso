package MultiLayerPerceptron;

import MultiLayerPerceptron.ActivationFunctions.Relu;
import MultiLayerPerceptron.ActivationFunctions.Sigmoid;
import MultiLayerPerceptron.ActivationFunctions.Tanh;
import MultiLayerPerceptron.GradientFunctions.AdaGrad;
import MultiLayerPerceptron.GradientFunctions.Adam;
import MultiLayerPerceptron.GradientFunctions.Momentum;
import MultiLayerPerceptron.GradientFunctions.SGD;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MLP {

    private static final String COMMA_DELIMITER = ",";
    private ArrayList<Layer> hiddenLayers = new ArrayList<Layer>();
    private Layer inputLayer;
    private Layer outputLayer;
    private int input;
    private int output;
    public GradientFunction gradientFunction;

    MLP(GradientFunction gradientFunction){
        this.gradientFunction = gradientFunction;
    }

    public void addInputLayer(int no_neurons){
        inputLayer = new Layer(no_neurons,gradientFunction);
        this.input = no_neurons;

    }

    public void initiateFeed(String[] input){
        for (int i = 0; i < inputLayer.no_neurons ; i++) {
            inputLayer.neurons[i].y_out = Float.parseFloat(input[i]);
        }
        inputLayer.feedForward();
    }

    public void addOutputLayer(int no_neurons, ActivationFunction activationFunction){
        outputLayer = new Layer(no_neurons, activationFunction, gradientFunction);
        this.output = no_neurons;
    }

    public void addHiddenLayer(int no_neurons, ActivationFunction activationFunction){
        hiddenLayers.add(new Layer(no_neurons,activationFunction,gradientFunction));
    }



    public void generateModel(){
        double weightInitiator = hiddenLayers.get(0).activationFunction.initiateWeight(input,output);

        inputLayer.generateNeurons(hiddenLayers.get(0),weightInitiator);
        int countHiddenLayer = hiddenLayers.size();
        for (int i = 0; i < countHiddenLayer-1 ; i++) {
            hiddenLayers.get(i).generateNeurons(hiddenLayers.get(i+1),weightInitiator);
        }
        hiddenLayers.get(countHiddenLayer-1).generateNeurons(outputLayer,weightInitiator);
        outputLayer.generateOutputNeuron();
    }

    public void fit(ArrayList<String []> input, double[] target,int epoch){
        int dataSize = input.size();
        for(int j = 0; j < epoch; j++){
            System.out.println("Epoch:-----------------------------"+j);
            for (int i = 0; i < dataSize ; i++) {
                initiateFeed(input.get(i));
                for(int k = 0; k < hiddenLayers.size(); k++){
                    hiddenLayers.get(k).feedForward();
                }

                if(outputLayer.neurons[0].checkOutput(target[i])==1){
                    for(int k = hiddenLayers.size()-1; k >= 0; k--){
                        hiddenLayers.get(k).backPropagate();
                    }
                    inputLayer.backPropagate(1);
                }
            }
        }

    }

    public void predict(String[] input){
        initiateFeed(input);
        for(int k = 0; k < hiddenLayers.size(); k++){
            hiddenLayers.get(k).feedForward();
        }
        System.out.print("Output"+outputLayer.neurons[0].y_out);
    }

    public static void main(String[] args) {

//        double input[][] = new double[4][];
//        input[0] = new double[]{0,0};
//        input[1] = new double[]{0,1};
//        input[2] = new double[]{1,0};
//        input[3] = new double[]{1,1};
        System.out.println(System.getProperty("user.dir"));
        File inputFile = new File("cancer.csv");
        File targetFile = new File("cancer_target.csv");
        ArrayList<String[]> input = new ArrayList<String[]>();
        Pattern p = Pattern.compile("\\d+");
        Matcher m;
        int noAttr=0;

        if(inputFile.exists()) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile), "UTF-8"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(COMMA_DELIMITER);
                    noAttr = values.length;
                    input.add(values);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            System.out.println("Input File not Found");
        }
        double[] target = new double[input.size()];

        if(targetFile.exists()) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(targetFile), "UTF-8"))) {
                String line;
                int z = 0;
                while ((line = br.readLine()) != null) {
                    m = p.matcher(line);
                    m.find();
                    target[z] = Float.parseFloat(m.group());

                    z++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            System.out.println("Target File not Found");
        }

        MLP model = new MLP(new AdaGrad());
        model.addInputLayer(noAttr);
        model.addHiddenLayer(noAttr,new Tanh());
        model.addHiddenLayer(noAttr+1,new Tanh());
        model.addHiddenLayer(noAttr+1,new Tanh());
        model.addHiddenLayer(noAttr,new Tanh());
        model.addOutputLayer(1,new Relu());
        model.generateModel();
        model.fit(input,target,5000);
        System.out.println("prediction");
        for(int i=0; i < input.size(); i++){
            model.predict(input.get(i));
            System.out.println("    target"+target[i]);
        }



    }



}
