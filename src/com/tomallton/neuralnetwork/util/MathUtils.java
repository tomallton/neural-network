package com.tomallton.neuralnetwork.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class MathUtils {

    public static double[] generate(int length, Supplier<Double> valueGenerator) {
        double[] arr = new double[length];
        for (int i = 0; i < length; i++) {
            arr[i] = valueGenerator.get();
        }
        return arr;
    }

    public static double[][] apply(double[][] a, Function<Double, Double> function) {
        for (int i = 0; i < a.length; i++) {
            a[i] = apply(a[i], function);
        }
        return a;
    }

    public static double[] apply(double[] a, Function<Double, Double> function) {
        for (int i = 0; i < a.length; i++) {
            a[i] = function.apply(a[i]);
        }
        return a;
    }

    public static int maxValueIndex(double... a) {
        double highest = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > highest) {
                highest = a[i];
                index = i;
            }
        }
        return index;
    }

    public static double[][] oneHot(double... a) {
        double[][] oneHot = new double[a.length][(int) max(a) + 1];
        for (int i = 0; i < oneHot.length; i++) {
            oneHot[i][(int) a[i]] = 1;
        }
        return oneHot;
    }

    public static double sum(double... a) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i];
        }
        return sum;
    }

    public static double max(double... a) {
        double max = Double.MIN_VALUE;
        for (double e : a) {
            max = Math.max(e, max);
        }
        return max;
    }

    public static double[] square(double... a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = Math.pow(a[i], 2);
        }
        return a;
    }

    public static double[][] add(double[][] a, double[][] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Length of arrays are not the same");
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = add(a[i], b[i]);
        }
        return a;
    }

    public static double[] add(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Length of arrays are not the same");
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = a[i] + b[i];
        }
        return a;
    }

    public static double[] subtract(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Length of arrays are not the same");
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = a[i] - b[i];
        }
        return a;
    }

    /**
     * Gets the cross product of 2 vectors.
     */
    public static double cross(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Length of arrays are not the same");
        }
        double cross = 0;
        for (int i = 0; i < a.length; i++) {
            cross += a[i] * b[i];
        }
        return cross;
    }

    /**
     * Normalizes a dataset, subtracting the mean from each value and diving by the standard deviation.
     */
    public static double[][] normalize(double[][] a) {
        double[] mean = mean(a);
        double[] standardDeviation = standardDeviation(a);

        for (int i = 0; i < a.length; i++) {
            double[] example = a[i];

            for (int j = 0; j < example.length; j++) {
                example[j] = (example[j] - mean[j]) / standardDeviation[j];
            }
        }

        return a;
    }

    public static double[] mean(double[][] a) {
        double[] mean = new double[a[0].length];

        for (double[] example : a) {
            for (int i = 0; i < example.length; i++) {
                mean[i] += example[i];
            }
        }

        for (int i = 0; i < mean.length; i++) {
            mean[i] /= a.length;
        }

        return mean;
    }

    public static double[] standardDeviation(double[][] a) {
        double[] mean = mean(a);
        double[] standardDeviation = new double[a[0].length];

        for (double[] example : a) {
            for (int i = 0; i < example.length; i++) {
                standardDeviation[i] += square(example[i] - mean[i]);
            }
        }

        for (int i = 0; i < mean.length; i++) {
            mean[i] /= a.length;
            mean[i] = Math.sqrt(mean[i]);
        }

        return standardDeviation;
    }

    public static double accuracy(double[][] yPredict, double[][] yActual) {
        if (yPredict.length != yActual.length) {
            throw new IllegalArgumentException("Length of arrays are not the same");
        }

        int correct = 0;

        for (int i = 0; i < yPredict.length; i++) {
            if (equals(yPredict[i], yActual[i])) {
                correct++;
            }
        }

        return correct / Double.valueOf(yPredict.length);
    }

    public static boolean equals(double[] a, double[] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    public static double square(double a) {
        return a * a;
    }

    public static double cube(double a) {
        return a * a * a;
    }

    public static Pair<Pair<double[][], double[][]>, Pair<double[][], double[][]>> trainTestSplit(double[][] X, double[][] y) {
        return trainTestSplit(X, y, 0.25);
    }

    public static Pair<Pair<double[][], double[][]>, Pair<double[][], double[][]>> trainTestSplit(double[][] X, double[][] y, double testProportion) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("Length of arrays are not the same");
        }

        List<Pair<double[], double[]>> examples = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            examples.add(Pair.of(X[i], y[i]));
        }
        Collections.shuffle(examples);

        int testSize = (int) Math.round(testProportion * examples.size());

        double[][] xTrain = new double[X.length - testSize][];
        double[][] xTest = new double[testSize][];

        double[][] yTrain = new double[xTrain.length][];
        double[][] yTest = new double[xTest.length][];

        for (int i = 0; i < xTest.length; i++) {
            xTest[i] = examples.get(i).getLeft();
            yTest[i] = examples.get(i).getRight();
        }
        for (int i = 0; i < xTrain.length; i++) {
            xTrain[i] = examples.get(testSize + i).getLeft();
            yTrain[i] = examples.get(testSize + i).getRight();
        }

        return Pair.of(Pair.of(xTrain, yTrain), Pair.of(xTest, yTest));
    }
}