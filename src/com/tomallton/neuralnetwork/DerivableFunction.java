package com.tomallton.neuralnetwork;

import java.util.function.Function;

public interface DerivableFunction extends Function<Double, Double> {

    Double derivative(Double x);

    public static DerivableFunction create(Function<Double, Double> fx, Function<Double, Double> derivative) {
        return new DerivableFunction() {
            @Override
            public Double apply(Double x) {
                return fx.apply(x);
            }

            @Override
            public Double derivative(Double x) {
                return derivative.apply(x);
            }
        };
    }
}