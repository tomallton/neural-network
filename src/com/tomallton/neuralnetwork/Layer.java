package com.tomallton.neuralnetwork;

import java.util.Arrays;
import java.util.function.Function;

import com.tomallton.neuralnetwork.util.MathUtils;
import com.tomallton.neuralnetwork.util.StringUtils;

public class Layer implements Function<double[], double[]> {
    private final double[][] weights;
    private final boolean bias;
    private final DerivableFunction activationFunction;

    public double[] lastOutputBeforeActivation, lastOutput;

    // initialize weights randomly
    public Layer(int neurons) {
        this(neurons, true);
    }

    public Layer(int neurons, boolean bias) {
        this(neurons, bias, ActivationFunction.NONE);
    }

    public Layer(int neurons, ActivationFunction activationFunction) {
        this(neurons, true, activationFunction);
    }

    public Layer(int neurons, boolean bias, ActivationFunction activationFunction) {
        this(new double[neurons + (bias ? 1 : 0)][], bias, activationFunction);
    }

    // pre-defined weights
    public Layer(double[][] weights) {
        this(weights, ActivationFunction.NONE);
    }

    public Layer(double[][] weights, boolean bias) {
        this(weights, bias, ActivationFunction.NONE);
    }

    public Layer(double[][] weights, ActivationFunction activationFunction) {
        this(weights, true, activationFunction);
    }

    public Layer(double[][] weights, boolean bias, ActivationFunction activationFunction) {
        for (double[] neuron : weights) {
            if (neuron != null && neuron.length > 0 && neuron.length != weights[0].length) {
                throw new IllegalArgumentException("Neurons have different number of weights");
            }
        }
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    @Override
    public double[] apply(double[] input) {
        if (input.length != getInputSize()) {
            throw new IllegalArgumentException(StringUtils.toString(input) + " does not have a length of " + getInputSize());
        }
        this.lastOutputBeforeActivation = new double[getOutputSize()];

        double[] output = new double[getOutputSize()];

        for (int weight = 0; weight < output.length; weight++) {
            for (int neuron = 0; neuron < weights.length; neuron++) {
                output[weight] += (neuron >= input.length && bias ? 1 : input[neuron]) * weights[neuron][weight];
            }
        }

        this.lastOutputBeforeActivation = output.clone();

        // apply activation function to output
        MathUtils.apply(output, activationFunction);

        this.lastOutput = output.clone();

        return output;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBias() {
        return bias ? weights[weights.length - 1] : null;
    }

    public boolean hasBias() {
        return bias;
    }

    public DerivableFunction getActivationFunction() {
        return activationFunction;
    }

    public int getInputSize() {
        return weights.length + (bias ? -1 : 0);
    }

    public int getOutputSize() {
        return weights[0].length;
    }

    public double[] getLastOutput() {
        return lastOutput;
    }

    public double[] getLastOutputBeforeActivation() {
        return lastOutputBeforeActivation;
    }

    @Override
    public String toString() {
        String weightsString = StringUtils.toString(bias ? Arrays.copyOfRange(weights, 0, weights.length - 1) : weights);
        return "Layer{weights=" + weightsString + ", " + (bias ? "bias=" + StringUtils.toString(getBias()) + ", " : "") + "activationFunction=" + activationFunction + "}";
    }
}