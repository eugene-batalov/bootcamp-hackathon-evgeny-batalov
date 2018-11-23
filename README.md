# bootcamp-hackathon-evgeny-batalov

This project is to test the idea of automatic log analyzing using ML algorithms.

Sample logs are placed in project resources folder, file for training algorithm 450k lines and 
file for testing 16k lines.


**Running project:**

Yse Gradle to run the project:

```
git clone https://github.com/trilogy-group/bootcamp-hackathon-evgeny-batalov.git loganalyzer
cd loganalyzer
./gradlew bootRun
```

This will run simple RNN <https://en.wikipedia.org/wiki/Recurrent_neural_network> training on train file and then
predicts errors in test log file. It runs 100 epochs, each epoch is about 5 minutes on core i7 CPU.

Below is sample output:
```$xslt
[main] INFO org.nd4j.linalg.factory.Nd4jBackend - Loaded [CpuBackend] backend
[main] INFO org.nd4j.nativeblas.NativeOpsHolder - Number of threads used for NativeOps: 4
[main] INFO org.nd4j.nativeblas.Nd4jBlas - Number of threads used for BLAS: 4
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Backend used: [CPU]; OS: [Windows 10]
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Cores: [8]; Memory: [5.3GB];
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Blas vendor: [MKL]
[main] INFO org.deeplearning4j.nn.multilayer.MultiLayerNetwork - Starting MultiLayerNetwork with WorkspaceModes set to [training: ENABLED; inference: ENABLED], cacheMode set to [NONE]
Epoch 0
[main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 0 is 2638796.5
real error line numbers: [1008, 1016, 1017, 3677, 7614, 7616]
predicted error line numbers: [439, 500, 2436, 2648, 2761, 2872, 3043, 3050, 3189, 3927, 4139, 4816, 4851, 4866, 4966, 5584, 6655, 6835, 8378, 8744, 9060, 9262, 9330, 10637, 10782, 10961, 10970, 11445, 11589, 11778, 12494, 12549, 12685, 13059, 13130, 13205, 13515, 14634, 15060, 15270, 15675, 15994]
Epoch done in 311 seconds
...
Epoch 26
[main] INFO org.deeplearning4j.optimize.listeners.ScoreIterationListener - Score at iteration 26 is 1641117.5
real error line numbers: [1008, 1016, 1017, 3677, 7614, 7616]
predicted error line numbers: [313, 385, 1060, 2126, 2652, 2921, 3204, 3457, 4898, 5298, 5932, 6319, 10769, 11448, 11933, 12192, 13272, 14715, 15902]
Epoch done in 369 seconds
```

In all, current implementation is not production ready but the idea definitely is interesting and worth further dig in.
