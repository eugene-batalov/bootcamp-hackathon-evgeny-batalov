package com.aurea.loganalyzer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import lombok.Data;
import lombok.experimental.UtilityClass;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

@Data
public class LogParser {

    private DataSet dataSet;
    private List<Integer> hashesList;
    private List<Integer> testList;

    public LogParser(Path trainingDataFilePath, Path testDataFilePath) throws IOException {
        dataSet = loadFromTextFile(trainingDataFilePath);
        testList = getTestList(testDataFilePath);
    }

    public DataSet loadFromTextFile(Path fileName) throws IOException {
        List<Integer> integers = Files.readAllLines(fileName).stream().map(log -> parseLogRecord(log, false))
                .collect(Collectors.toList());
        hashesList = new HashSet<>(integers).stream().collect(Collectors.toList());
        INDArray input = Nd4j.zeros(1, hashesList.size(), integers.size());
        INDArray labels = Nd4j.zeros(1, hashesList.size(), integers.size());
        // loop through our sample-sentence
        int samplePos = 0;
        for (int currentInt : integers) {
            // small hack: when currentInt is the last, take the first char as
            // nextChar - not really required. Added to this hack by adding a starter first character.
            int nextInt = integers.get((samplePos + 1) % (integers.size()));
            // input neuron for current-int is 1 at "samplePos"
            input.putScalar(new int[]{0, hashesList.indexOf(currentInt), samplePos}, 1);
            // output neuron for next-int is 1 at "samplePos"
            labels.putScalar(new int[]{0, hashesList.indexOf(nextInt), samplePos}, 1);
            samplePos++;
        }
        return new DataSet(input, labels);
    }

    public Integer parseLogRecord(String log, boolean fromList) {
        String[] strings = log.trim().split("\\s+");
        if (Objects.isNull(strings) || strings.length < 11) {
            // type: other
            return 0;
        }
        try {
            LocalDate.parse(strings[1]);
        } catch (DateTimeParseException exception) {
            return 0;
        }
        String level = strings[3];
        if (level.equals("ERROR")) {
            return 1;
        }
        String s = strings[3] + strings[8] + strings[10];
        int hashCode = s.hashCode();
        return !fromList || hashesList.contains(hashCode)? hashCode : 0;
    }

    public List<Integer> getTestList(Path fileName) throws IOException {
        return Files.readAllLines(fileName).stream().map(log -> parseLogRecord(log, true))
                .collect(Collectors.toList());
    }
}
