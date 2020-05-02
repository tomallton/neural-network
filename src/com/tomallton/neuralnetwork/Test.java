package com.tomallton.neuralnetwork;

import java.util.Arrays;
import java.util.List;

import com.tomallton.neuralnetwork.util.FileUtils;
import com.tomallton.neuralnetwork.util.MathUtils;
import com.tomallton.neuralnetwork.util.Pair;
import com.tomallton.neuralnetwork.util.StringUtils;

public class Test {

    public static void main(String[] args) {
        List<String> lines = FileUtils.readFile("data/cancer.csv");

        double[][] X = new double[lines.size()][];
        double[][] y = new double[lines.size()][1];

        for (int i = 0; i < lines.size(); i++) {
            String[] fields = lines.get(i).split(",");

            // skip id and classification field
            X[i] = Arrays.stream(Arrays.copyOfRange(fields, 2, fields.length)).mapToDouble(Double::valueOf).toArray();
            // label as 1 if positive for cancer (malignant)
            y[i] = new double[] { fields[1].equals("M") ? 1 : 0 };
        }

        // normalize data
        MathUtils.normalize(X);

        NeuralNetwork model = new NeuralNetwork(new Layer(X[0].length, ActivationFunction.SIGMOID), new Layer(25, ActivationFunction.SIGMOID), new Layer(15), new Layer(1));

        Pair<Pair<double[][], double[][]>, Pair<double[][], double[][]>> trainTestSplit = MathUtils.trainTestSplit(X, y);

        double[][] xTrain = trainTestSplit.getLeft().getLeft();
        double[][] yTrain = trainTestSplit.getLeft().getRight();
        double[][] xTest = trainTestSplit.getRight().getLeft();
        double[][] yTest = trainTestSplit.getRight().getRight();

        model.train(xTrain, yTrain, 0.1, 100);

        double[][] yPredict = model.predict(xTest);

        // round prediction to nearest whole value
        MathUtils.apply(yPredict, a -> a >= 0.5 ? 1D : 0D);

        // print accuracy
        double accuracy = MathUtils.accuracy(yPredict, yTest);
        System.out.println("Accuracy: " + StringUtils.formatPercentage(accuracy));
    }

    public static void calculationsTest() {

        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        NeuralNetwork model = new NeuralNetwork(new Layer(new double[][] { { 0.15, 0.25 }, { 0.2, 0.3 }, { 0.35, 0.35 } }, ActivationFunction.SIGMOID),
                new Layer(new double[][] { { 0.4, 0.5 }, { 0.45, 0.55 }, { 0.6, 0.6 } }, ActivationFunction.SIGMOID));

        model.train(new double[][] { { 0.05, 0.1 } }, new double[][] { { 0.01, 0.99 } }, 0.5);

        System.out.println(model);
    }
}