import java.util.Random;
public class bitNeuralNetwork {
    private double[][] weightsInputHidden; private double[] weightsHiddenOutput; private double[] hiddenLayer; private double learningRate;
    public bitNeuralNetwork(int inputSize, int hiddenSize, double learningRate) {
        this.weightsInputHidden = new double[inputSize][hiddenSize];
        this.weightsHiddenOutput = new double[hiddenSize];
        this.hiddenLayer = new double[hiddenSize];
        this.learningRate = learningRate;
        newWeights();
    }
    private void newWeights() {
        Random random = new Random();
        for (int i = 0; i < weightsInputHidden.length; i++) {
            for (int j = 0; j < weightsInputHidden[0].length; j++) {
                weightsInputHidden[i][j] = random.nextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < weightsHiddenOutput.length; i++) {
            weightsHiddenOutput[i] = random.nextDouble() * 2 - 1;
        }
    }
    private double activation(double x) {
        return 1 / (1 + Math.exp(-x)); //sigmoid but probably gonna add like relu
    }
    private double ddxactivation(double x) {
        return x * (1 - x);
    }
    public double predict(double[] inputs) {
        for (int j = 0; j < hiddenLayer.length; j++) {
            hiddenLayer[j] = 0;
            for (int i = 0; i < inputs.length; i++) {
                hiddenLayer[j] += inputs[i] * weightsInputHidden[i][j];
            }
            hiddenLayer[j] = activation(hiddenLayer[j]);
        }

        double output = 0;
        for (int j = 0; j < hiddenLayer.length; j++) {
            output += hiddenLayer[j] * weightsHiddenOutput[j];
        }
        return activation(output);
    }
    public void train(double[][] inputs, double[] targets, int max) {
        for (int num = 0; num < max; num++) {
            for (int sample = 0; sample < inputs.length; sample++) {
                //forward propagate
                double[] input = inputs[sample];
                double target = targets[sample];

                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenLayer[j] = 0;
                    for (int i = 0; i < input.length; i++) {
                        hiddenLayer[j] += input[i] * weightsInputHidden[i][j];
                    }
                    hiddenLayer[j] = activation(hiddenLayer[j]);
                }

                double output = 0;
                for (int j = 0; j < hiddenLayer.length; j++) {
                    output += hiddenLayer[j] * weightsHiddenOutput[j];
                }
                //backpropagate using gradient descent (a bit simplified)
                output = activation(output);
                double error = target - output;
                double outputGradient = error * ddxactivation(output);

                for (int j = 0; j < weightsHiddenOutput.length; j++) {
                    weightsHiddenOutput[j] += learningRate * outputGradient * hiddenLayer[j];
                }

                for (int j = 0; j < hiddenLayer.length; j++) {
                    double hiddenError = outputGradient * weightsHiddenOutput[j];
                    double hiddenGradient = hiddenError * ddxactivation(hiddenLayer[j]);

                    for (int i = 0; i < input.length; i++) {
                        weightsInputHidden[i][j] += learningRate * hiddenGradient * input[i];
                    }
                }
            }
        }
    }
    public static void main(String[] args) {
        bitNeuralNetwork nn = new bitNeuralNetwork(3, 4, 0.1); //idk why i did this i shouldve set the hidden and input size
        double[][] inputs = { //training data idk imma add a bitmap i think
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1}
        };
        double[] targets = {0, 1, 0, 1, 0, 1, 0, 1};
        nn.train(inputs, targets, 10000);
        for (double[] input : inputs) {
            System.out.println("input array: " + java.util.Arrays.toString(input) + " predicted output: " + String.format("%.3f", nn.predict(input)));
        }
        double totalError = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double predicted = nn.predict(input);
            double actual = targets[i];
            totalError += Math.pow(predicted - actual, 2);
        }
        double meanSquaredError = totalError / inputs.length;
        System.out.println("accuracy (mse): "+ (String.format("%.5f", 100-meanSquaredError*100)));
    }
}
