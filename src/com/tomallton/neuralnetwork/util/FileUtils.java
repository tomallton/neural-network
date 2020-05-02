package com.tomallton.neuralnetwork.util;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FileUtils {

    public static List<String> readFile(File file) {
        return readFile(file.getPath());
    }

    public static List<String> readFile(String fileName) {
        try (Stream<String> stream = Files.lines(Paths.get(fileName))) {
            return stream.collect(Collectors.toList());
        } catch (IOException exception) {
            exception.printStackTrace();
        }
        return null;
    }

}