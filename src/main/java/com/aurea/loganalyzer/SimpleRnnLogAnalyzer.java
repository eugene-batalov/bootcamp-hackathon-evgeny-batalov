package com.aurea.loganalyzer;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;


public class SimpleRnnLogAnalyzer {

    // RNN dimensions
    private static final int HIDDEN_LAYER_WIDTH = 25;
    private static final int HIDDEN_LAYER_CONT = 2;
    private static final Random r = new Random(7894);

    public static void main(String[] args) throws URISyntaxException, IOException {

        // some common parameters
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.seed(123);
        builder.biasInit(0);
        builder.miniBatch(false);
        builder.updater(new RmsProp(0.001));
        builder.weightInit(WeightInit.XAVIER);

        ListBuilder listBuilder = builder.list();

        ClassLoader classLoader = SimpleRnnLogAnalyzer.class.getClassLoader();
        Path trainLogsPath = Paths.get(classLoader.getResource("train000000.logs").toURI());
        Path testLogsPath = Paths.get(classLoader.getResource("test000000.logs").toURI());
        LogParser logParser = new LogParser(trainLogsPath, testLogsPath);
        DataSet trainingData = logParser.getDataSet();
        int shape_size = trainingData.getFeatures().shape()[1];
        // first difference, for rnns we need to use LSTM.Builder
        for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
            LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? shape_size : HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
            // adopted activation function from LSTMCharModellingExample
            // seems to work well with RNNs
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        // we need to use RnnOutputLayer for our RNN
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
        // softmax normalizes the output neurons, the sum of all outputs is 1
        // this is required for our sampleFromDistribution-function
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
        outputLayerBuilder.nOut(shape_size);
        listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

        // finish builder
        listBuilder.pretrain(false);
        listBuilder.backprop(true);

        // create network
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        Random rng = new Random(12345);

        List<Integer> testList = logParser.getTestList();
        List<Integer> hashesList = logParser.getHashesList();
        // some epochs
        for (int epoch = 0; epoch < 100; epoch++) {

            System.out.println("Epoch " + epoch);

            // train the data
            Instant start = Instant.now();
            net.fit(trainingData);

            // clear current stance from the last example
            net.rnnClearPreviousState();

            List<Integer> testErrorIndexes = new ArrayList<>();
            List<Integer> predictedErrorIndexes = new ArrayList<>();

            // now the net should guess
            for (int lineNumber = 0; lineNumber < testList.size() - HIDDEN_LAYER_WIDTH - 1; lineNumber++) {

                // put the first character into the rrn as an initialisation
                INDArray testInit = Nd4j.zeros(1, shape_size, HIDDEN_LAYER_WIDTH);
                Integer first = testList.get(0);
                if (first == 1) {
                    testErrorIndexes.add(0);
                }
                for (int i = 0; i < HIDDEN_LAYER_WIDTH; i++) {
                    testInit.putScalar(new int[]{0, hashesList.indexOf(testList.get(lineNumber + i)), i}, 1);
                }

                // run one step -> IMPORTANT: rnnTimeStep() must be called, not
                // output()
                // the output shows what the net thinks what should come next
                INDArray output = net.rnnTimeStep(testInit);
                output = output.tensorAlongDimension((int)output.size(2)-1,1,0);
                // first process the last output of the network to a concrete
                // neuron, the neuron with the highest output has the highest
                // chance to get chosen
                // INDArray exec = Nd4j.getExecutioner().exec(new IMax(output), 1);
                //int sampledIntIdx = exec.getInt(0);
                double[] outputProbDistribution = new double[shape_size];
                for (int i = 0; i < shape_size; i++) {
                    outputProbDistribution[i] = output.getDouble(0, i);
                }
                int sampledIntIdx = sampleFromDistribution(outputProbDistribution, rng);

                // System.out.println("sampledIntIdx: " + sampledIntIdx);

                if (sampledIntIdx == 1) {
                    predictedErrorIndexes.add(lineNumber);
                }

                if (testList.get(lineNumber + HIDDEN_LAYER_WIDTH) == 1) {
                    testErrorIndexes.add(lineNumber);
                }
            }
            System.out.println("real error line numbers: " + testErrorIndexes);
            System.out.println("predicted error line numbers: " + predictedErrorIndexes);
            long seconds = Duration.between(start, Instant.now()).getSeconds();
            System.out.println("Epoch done in " + seconds + " seconds");
        }
    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution( double[] distribution , Random rng){

        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}
