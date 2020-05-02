package com.tomallton.neuralnetwork;

import java.util.function.Function;

import com.tomallton.neuralnetwork.util.StringUtils;

public enum ActivationFunction implements DerivableFunction {
    NONE(x -> x, x -> 1D),
    RELU(x -> Math.max(x, 0), x -> x > 0 ? 1D : 0D),
    LEAKY_RELU(x -> Math.max(0.1 * x, x), x -> x > 0 ? 1D : 0.1D),
    SIGMOID(x -> 1 / (1 + Math.pow(Math.E, -x)), x -> {
        double fx = 1 / (1 + Math.pow(Math.E, -x));
        return fx * (1 - fx);
    }),
    TANH(x -> Math.tanh(x), x -> 1 / Math.pow(Math.cosh(x), 2)),
    STEP(x -> x > 0 ? 1D : 0D, x -> 0D);

    private final Function<Double, Double> function, derivative;

    ActivationFunction(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    @Override
    public Double apply(Double x) {
        return function.apply(x);
    }

    @Override
    public Double derivative(Double x) {
        return derivative.apply(x);
    }

    @Override
    public String toString() {
        return StringUtils.capitalise(name());
    }
}