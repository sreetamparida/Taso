# Taso
Java implementation of Multilayer Perceptron

## Implementation

### Modules
It basically has three main modules `MLP`, `Gradient Function` and `Activation Function`.

**MLP:** It is the main driver class that helps create the model to fit and predict.

**Activation Function:** This java implementation currently supports these following Activation Functions
- Relu
- LeakyRelu
- Tanh
- Sigmoid

**Gradient Function:** This java implementation currently supports these following Gradient Functions
- AdaGrad
- Adam
- Momentum
- RMSProp
- SGD

### Environment Required

- [x] Windows, Linux or MacOS
- [x] Java 1.8

## USING THIS PROJECT

After all the pre-requisites mentioned above is installed on the machine then follow the steps provided below.

1. Clone this repository `https://github.com/sreetamparida/Taso.git` and open `MLP.java`.
2. Provide the path to Training and Tesing dataset as shown below.

```java
File inputFile = new File("cancer.csv");
File targetFile = new File("cancer_target.csv");
```
3. Create the model with your specified `Gradient Function`, `Activation Function` and `Hidden Layers`.
4. Specify the Gradient Function while creating the model object by adding the Gradient Function object as parameter.
5. For adding `Hidden Layers` use `addHiddenLayer()` function.
6. Specify the Activation Function for the layer by adding the Activation Function object as parameter.
7. Then perform the `fit()` operation by providing number of `epochs`.
8. Use the `predict()` function to specify the target data.

An implementation of a **model** with 
- **Gradient Function** - AdaGrad
- **Activation Function** - TanH in the hidden layers and Relu at the output layer
- **Number of Hidden Layers** - 4
```java
        MLP model = new MLP(new AdaGrad());
        model.addInputLayer(noAttr);
        model.addHiddenLayer(noAttr,new Tanh());
        model.addHiddenLayer(noAttr+1,new Tanh());
        model.addHiddenLayer(noAttr+1,new Tanh());
        model.addHiddenLayer(noAttr,new Tanh());
        model.addOutputLayer(1,new Relu());
        model.generateModel();
        model.fit(input,target,5000);
```
9. Execute the comand `javac MLP.java` to get your desired predictions.


