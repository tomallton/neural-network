package com.tomallton.neuralnetwork.util;

import java.util.Objects;

/**
 * Stores two objects together.
 *
 * @param <L> The type of the first object.
 * @param <R> The type of the second object.
 */
public class Pair<L, R> {
    private final L left;
    private final R right;

    private Pair(L left, R right) {
        this.left = left;
        this.right = right;
    }

    public L getLeft() {
        return left;
    }

    public R getRight() {
        return right;
    }

    public boolean hasLeft() {
        return left != null;
    }

    public boolean hasRight() {
        return right != null;
    }

    public boolean isFull() {
        return left != null && right != null;
    }

    @Override
    public String toString() {
        return "(" + left + ", " + right + ")";
    }

    @Override
    public int hashCode() {
        return Objects.hash(left, right);
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        if (other == this) {
            return true;
        }
        if (!(other instanceof Pair)) {
            return false;
        }
        Pair<?, ?> pair = (Pair<?, ?>) other;

        return left.equals(pair.getLeft()) && right.equals(pair.getRight());
    }

    public static <L, R> Pair<L, R> of(L left, R right) {
        return new Pair<>(left, right);
    }
}