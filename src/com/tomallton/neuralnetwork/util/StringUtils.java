package com.tomallton.neuralnetwork.util;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.stream.Collectors;

public class StringUtils {

    public static String capitalise(String string) {
        StringBuilder out = new StringBuilder(string.toLowerCase());
        out.setCharAt(0, Character.toUpperCase(out.charAt(0)));
        for (int i = 2; i < string.length(); i++) {
            if (out.charAt(i - 1) == ' ') {
                out.setCharAt(i, Character.toUpperCase(out.charAt(i)));
            }
        }
        return out.toString();
    }

    public static String formatPercentage(double percentage) {
        return new DecimalFormat("0.0").format(percentage * 100) + " %";
    }

    public static String toString(double[][][] a) {
        return "[" + String.join(", ", Arrays.stream(a).map(StringUtils::toString).collect(Collectors.toList())) + "]";
    }

    public static String toString(double[][] a) {
        return "[" + String.join(", ", Arrays.stream(a).map(StringUtils::toString).collect(Collectors.toList())) + "]";
    }

    public static String toString(double[] a) {
        return "[" + String.join(", ", Arrays.stream(a).mapToObj(String::valueOf).collect(Collectors.toList())) + "]";
    }

    public static String toString(int[][][] a) {
        return "[" + String.join(", ", Arrays.stream(a).map(StringUtils::toString).collect(Collectors.toList())) + "]";
    }

    public static String toString(int[][] a) {
        return "[" + String.join(", ", Arrays.stream(a).map(StringUtils::toString).collect(Collectors.toList())) + "]";
    }

    public static String toString(int[] a) {
        return "[" + String.join(", ", Arrays.stream(a).mapToObj(String::valueOf).collect(Collectors.toList())) + "]";
    }
}