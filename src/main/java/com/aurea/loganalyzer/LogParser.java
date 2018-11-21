package com.aurea.loganalyzer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import lombok.experimental.UtilityClass;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

@UtilityClass
public class LogParser {

    public DataSet loadFromTextFile(Path fileName) throws IOException {
        List<Integer> integers = Files.readAllLines(fileName).stream().map(LogParser::parseLogRecord)
                .collect(Collectors.toList());
        Integer size = new HashSet<>(integers).size();
        INDArray input = Nd4j.zeros(1, 1, size);
        INDArray labels = Nd4j.zeros(1, 1, size);
        // loop through our sample-sentence
        int samplePos = 0;
        for (int currentInt : integers) {
            // small hack: when currentChar is the last, take the first char as
            // nextChar - not really required. Added to this hack by adding a starter first character.
            int nextInt = integers.get((samplePos + 1) % (size));
            // input neuron for current-int is 1 at "samplePos"
            input.putScalar(new int[]{0, currentInt, samplePos}, 1);
            // output neuron for next-int is 1 at "samplePos"
            labels.putScalar(new int[]{0, nextInt, samplePos}, 1);
            samplePos++;
        }
        DataSet trainingData = new DataSet(input, labels);
        return trainingData;
    }

    public Integer parseLogRecord(String log) {
        String[] strings = log.split(" ");
        if (Objects.isNull(strings) || strings.length < 20) {
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
        return (strings[3] + strings[9] + strings[19]).hashCode();
    }
}
