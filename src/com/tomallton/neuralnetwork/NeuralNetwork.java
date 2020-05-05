package com.tomallton.neuralnetwork;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.tomallton.neuralnetwork.util.MathUtils;

public class NeuralNetwork implements Function<double[], double[]> {
    private final Layer[] layers;

    public NeuralNetwork(List<Layer> layers) {
        this(layers.toArray(Layer[]::new));
    }

    public NeuralNetwork(Layer... layers) {
        // initialize layer weights
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            for (int j = 0; j < layer.getWeights().length; j++) {
                if (layer.getWeights()[j] == null || layer.getWeights()[j].length == 0) {
                    // last layer has same input and output size
                    layer.getWeights()[j] = MathUtils.generate(i < layers.length - 1 ? layers[i + 1].getInputSize() : layer.getInputSize(), Math::random);
                }
            }
        }

        this.layers = layers;
    }

    public void train(double[][] X, double[][] y, double learningRate, int epochs) {
        for (int i = 0; i < epochs; i++) {
            train(X, y, learningRate);
        }
    }

    public double[] train(double[][] X, double[][] y, double learningRate) {
        double[] totalErrors = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            double[] yPredict = predict(X[i]);
            double[] yTarget = y[i];

            double error = 0.5 * MathUtils.sum(MathUtils.square(MathUtils.subtract(yTarget.clone(), yPredict)));

            totalErrors[i] = error;

            if (error != 0) {
                double[][][] errors = new double[layers.length + 1][][];
                double[][][] weightChanges = new double[layers.length][][];

                // add errors from output layers
                errors[layers.length] = new double[yPredict.length][1];
                for (int j = 0; j < yPredict.length; j++) {
                    errors[layers.length][j][0] = yPredict[j] - yTarget[j];
                }

                // backpropagate errors starting from the last layer
                for (int layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
                    Layer layer = layers[layerIndex];

                    // if first hidden layer, last layer input is the actual input
                    double[] lastInput = layerIndex == 0 ? X[i] : layers[layerIndex - 1].getLastOutput();

                    errors[layerIndex] = new double[layer.getWeights().length][layer.getOutputSize()];
                    weightChanges[layerIndex] = new double[layer.getWeights().length][layer.getOutputSize()];

                    for (int neuronIndex = 0; neuronIndex < layer.getWeights().length; neuronIndex++) {
                        double gradientNetToWeight = layer.hasBias() && neuronIndex >= lastInput.length ? 1 : lastInput[neuronIndex];

                        for (int weightIndex = 0; weightIndex < layer.getWeights()[neuronIndex].length; weightIndex++) {
                            // sum errors from the previous layer
                            double gradientOutputToNet = layer.getActivationFunction().derivative(layer.getLastOutputBeforeActivation()[weightIndex]);
                            double gradientErrorToOutput = MathUtils.sum(errors[layerIndex + 1][weightIndex]);

                            errors[layerIndex][neuronIndex][weightIndex] = gradientOutputToNet * gradientErrorToOutput * layer.getWeights()[neuronIndex][weightIndex];
                            weightChanges[layerIndex][neuronIndex][weightIndex] = -learningRate * gradientErrorToOutput * gradientOutputToNet * gradientNetToWeight;
                        }
                    }
                }

                // apply weight changes
                for (int layerIndex = 0; layerIndex < weightChanges.length; layerIndex++) {
                    MathUtils.add(layers[layerIndex].getWeights(), weightChanges[layerIndex]);
                }
            }
        }

        return totalErrors;
    }

    @Override
    public double[] apply(double[] x) {
        return predict(x);
    }

    public double[][] predict(double[][] X) {
        double[][] output = new double[X.length][];

        for (int i = 0; i < X.length; i++) {
            output[i] = predict(X[i]);
        }

        return output;
    }

    public double[] predict(double... x) {
        double[] output = x.clone();

        for (Layer layer : layers) {
            output = layer.apply(output);
        }

        return output;
    }

    public Layer[] getLayers() {
        return layers;
    }

    public int getInputSize() {
        return layers[0].getInputSize();
    }

    public int getOutputSize() {
        return layers[layers.length - 1].getOutputSize();
    }

    @Override
    public String toString() {
        return "NeuralNetwork{layers=" + String.join(", ", Stream.of(layers).map(Object::toString).collect(Collectors.toList())) + "}";
    }
}