'''

Most of this code is duct taped together from various sources online. I don't know much about machine learning, but I wanted to learn as much as possible by recoding a simple neural network.

<3 farhan r. 11/17/17


'''


import numpy as np
import math
import random


# Sigmoid transfer function
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

# Derivative of a sigmoid.
def dsigmoid(y):
    return y * (1.0 - y);

# Tanh is probably better than sigmoid.
def tanh(x):
    return math.tanh(x);

# Derivative for tanh sigmoid.
def dtanh(y):
    return 1 - y*y


class MLP_NeuralNetwork(object):
    '''
    This is a multi layer perceptron network. There are three layers: input,
    output, and hidden layers.

    The algorithm has been generalized to be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    '''

    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):

        '''
        parameter: input = number of input neurons.
        parameter: hiden = number of hidden neurons.
        parameter: output = number of output neurons.
        '''

        # Initialize the parameters.
        self.iterations = iterations
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.learning_rate = learning_rate

        # Initialize the arrays.
        self.input = input + 1 # Add one so there is a bias.
        self.hidden = hidden
        self.output = output


        # Set up array of 1s for activation.
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # Create randomized weights
        input_range = 1.0 / self.input ** (1/2)
        output_range = 1.0 / self.hidden ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden));
        self.wo = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output))


        # These are the temporary values that get changed at each iteration.
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))


    def feedForward(self, inputs):

        '''
        This is the feed forward algorithm. It loops over all the nodes in the hidden layer and then adds togeter with all the outputs of the input * their weights.

        param inputs = input data
        return = updated ativation output vector.


        '''

        # Error handling.
        if(len(inputs) != self.input-1):
            raise ValueError('Wrong number of inputs bro!')

        # Input Activations
        for i in range(self.input - 1): ## We subtract 1 to ignore bias.
            self.ai[i] = inputs[i]


        # Hidden Activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)


        # Output Activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]

            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropogate(self, targets):
        '''

        1) For the Output Layer.

            a) Calculates the difference between the output and the target value.

            b) Get the slope also known as the Derivative of the sigmoid function to determine how much of the weights we need to change.

            c) Update the weights for every node based on the learning rate and the sigmoid derivative.

        2) For the Hidden Layer.

            a) Calculate the sum of the strength of each output multiplied by how much it has to change.

            b) get derivative to determine how much the weights need to change.

            c) change the weights based on the learning rate and derivative.

        parameter targets = y values
        parameter N = learning rate
        return = updated weights


        '''

        if(len(targets) != self.output):
            raise ValueError('Wrong number of targets! SMH')

        # Calculate the error terms for the output.
        # Delta tells you which direction to change the weights
        output_deltas = [0.0] * self.output

        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error = -(targets[k] - self.ao[k])
                output_deltas[k] = dsigmoid(self.ao[k]) * error

        # Calculate the terms for hidden
        # Delta tells you which direction to change the weights.
        hidden_deltas = [0.0] * self.hidden

        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error


        # Update the weights connecting the hidden to the output.
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        # Update the weights connecting the input to the hidden.
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        # Calculate the error.
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self,patterns):
        '''
        This will print out the targets next to the predictions.
        '''
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))


    def train(self,patterns):

        # N : Learning rate.
        for i in range(self.iterations):


            error = 0.0
            random.shuffle(patterns)

            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropogate(targets)
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            if i % 10 == 0:
                print('error %-.5f' % error)

            #Learning Rate Decay.
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))

    def predict(self, X):
        '''
        Return list of predictions after training the algorithm.
        '''
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))

        return predictions

## Load data with this simple function.
def load_data():
    data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')

    # first ten values are the one hot encoded y (target) values
    y = data[:,0:10]
    #y[y == 0] = -1 # if you are using a tanh transfer function make the 0 into -1
    #y[y == 1] = .90 # try values that won't saturate tanh

    data = data[:,10:] # x data
    #data = data - data.mean(axis = 1)
    data -= data.min() # scale the data so values are between 0 and 1
    data /= data.max() # scale

    out = []
    print data.shape

    # populate the tuple list with the data
    for i in range(data.shape[0]):
        fart = list((data[i,:].tolist(), y[i].tolist())) # don't mind this variable name
        out.append(fart)

    return out


def demo():
    '''
    Run NN demo from digit recognition dataset provided by SKITLearn.
    '''
    X = load_data();

    NN = MLP_NeuralNetwork(64,100,10, iterations = 50, learning_rate = 0.5, momentum = 0.5, rate_decay = 0.01)

    NN.train(X)

    NN.test(X)


if __name__ == "__main__":
    demo()
