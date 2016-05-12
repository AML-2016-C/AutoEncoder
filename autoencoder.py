import numpy as np
import pickle, random, pdb

def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.01, reg_lambda=0.01):
        """
        Initialize the network with input, output sizes, weights and biases
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :param reg_lambda: regularization rate lambda
        :return: None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01 # Weight matrix for input to hidden
        self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01 # Weight matrix for hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.output_dim, 1)) # output bias
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # pdb.set_trace()
        h_a = np.tanh(np.dot(self.Wxh, np.reshape(X,(len(X),1))) + self.bh)
        #ys = np.exp(np.dot(self.Why, h_a) + self.by)
        #probs = ys/np.sum(ys)
        probs = np.dot(self.Why,h_a) + self.by
        return h_a, probs

    def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
        """
        Add regularization terms to the weights
        :param dWhy: weight derivative from hidden to output
        :param dWxh: weight derivative from input to hidden
        :param Why: weights from hidden to output
        :param Wxh: weights from input to hidden
        :return: dWhy, dWxh
        """
        dWhy += self.reg_lambda * Why
        dWxh += self.reg_lambda * Wxh
        return dWhy, dWxh

    def _update_parameter(self, dWxh, dbh, dWhy, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        self.Wxh += -self.learning_rate * dWxh
        self.bh += -self.learning_rate * dbh
        self.Why += -self.learning_rate * dWhy
        self.by += -self.learning_rate * dby

    def _back_propagation(self, X, t, h_a, probs):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        dWxh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dy = np.copy(probs)
        #dy[t] -= 1
        kd=probs.reshape(len(t))
        for i in range(len(X)):
            meanError= (kd[i] - t[i])
            dy[i]=meanError
        #print type(dy)
        dWhy = np.dot(dy, h_a.T)
        dby += dy
        # pdb.set_trace()
        dh = np.dot(self.Why.T, dy)  # backprop into h
        dhraw = (1 - h_a * h_a) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        # pdb.set_trace()
        dWxh += np.dot(dhraw, np.reshape(X, (len(X), 1)).T)
        return dWxh, dWhy, dbh, dby

    def _calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
        """
        Calculate the smoothened loss over the set of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: smooth loss
        """
        if regularizer_type == 'L2':
            # Add regulatization term to loss
            loss += self.reg_lambda/2 * (np.sum(np.square(self.Wxh)) + np.sum(np.square(self.Why)))
            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    def train(self, inputs, targets, validation_data, num_epochs, regularizer_type=None):
        """
        Trains the network by performing forward pass followed by backpropagation
        :param inputs: list of training inputs
        :param targets: list of corresponding training targets
        :param validation_data: tuple of (X,y) where X and y are inputs and targets
        :param num_epochs: number of epochs for training the model
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: None
        """
        for k in xrange(num_epochs):
            loss = 0
            for i in xrange(len(inputs)):
                # Forward pass
                h_a, probs = self._feed_forward(inputs[i])
                #pdb.set_trace()
                #print probs
                #print inputs[i]
                #print "probs = ",probs.shape
                #print targets[i].shape
                error = 0
                for r in range(len(targets[i])):
                    meanError= (probs[r,0] - targets[i][r])**2
                    error+= meanError
                #print error

                #loss += -np.log(probs[targets[i], 0])
                loss += error/(2*len(targets[i]))
                #pdb.set_trace()
                # Backpropagation
                dWxh, dWhy, dbh, dby = self._back_propagation(inputs[i], targets[i], h_a, probs)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh, dbh, dWhy, dby)

            # validation using the validation data

            validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            print 'Validation'

            for i in xrange(len(validation_inputs)):
                # Forward pass
                h_a, probs = self._feed_forward(inputs[i])
                #loss += -np.log(probs[targets[i], 0])
                error = 0
                for r in range(len(targets[i])):
                    meanError= (probs[r,0] - targets[i][r])**2
                    error+= meanError



                #loss += -np.log(probs[targets[i], 0])
                loss += error/(2*len(targets[i]))

                # Backpropagation
                dWxh, dWhy, dbh, dby = self._back_propagation(inputs[i], targets[i], h_a, probs)

                if regularizer_type == 'L2':
                    dWhy, dWxh = self._regularize_weights(dWhy, dWxh, self.Why, self.Wxh)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh, dbh, dWhy, dby)

            if k%1 == 0:
                print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        h_a, probs = self._feed_forward(X)
        # return probs
        return np.argmax(probs)

    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

if __name__ == "__main__":

    hiddenSizeL1 = 16
    hiddenSizeL2 = 8
    '''auto_encoder1= NeuralNetwork(4,hiddenSizeL1,4)
    inputs = []
    targets = []
    for i in range(1000):
        num = random.randint(0,3)
        inp = np.zeros((4,))
        inp[num] = 1
        inputs.append(inp)
        targets.append(num)
    '''
    l=pickle.load(open("inputVectors.pkl"))
    auto_encoder1 = NeuralNetwork(768,hiddenSizeL1,768)
    inputs1=[]
    for key in l.keys():
        for j in range(len(l[key][:int(0.8*len(l[key]))])):
            vector=l[key][j]
            inputs1.append(vector)

    #inputs1=inputs
    #print type(inputs[0])
    print type(inputs1[0])
    val = int(0.9*len(inputs1))
    auto_encoder1.train(inputs1[:val], inputs1[:val], (inputs1[val:], inputs1[val:]), 10, regularizer_type='L2')
    inputs2 = []
    for i in range(len(inputs1)):
        h_a, probs = auto_encoder1._feed_forward(inputs1[i])
        h_a = h_a.reshape(hiddenSizeL1)
        #print type(h_a[1])
        inputs2.append(h_a)

    auto_encoder2 = NeuralNetwork(hiddenSizeL1,hiddenSizeL2,hiddenSizeL1)
    auto_encoder2.train(inputs2[:val], inputs2[:val], (inputs2[val:], inputs2[val:]), 10, regularizer_type='L2')
    pickle.dump(auto_encoder1,open("autoencoder1.pkl","wb"))
    pickle.dump(auto_encoder2,open("autoencoder2.pkl","wb"))
