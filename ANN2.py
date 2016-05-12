from __future__ import division
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
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim, learning_rate=0.01, reg_lambda=0.01):
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
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.Wxh1 = np.random.randn(self.hidden_size1, self.input_dim) * 0.01 # Weight matrix for input to hidden1
        self.Wh1h2 = np.random.randn(self.hidden_size2, self.hidden_size1) * 0.01 # Weight matrix for hidden1 to hidden2
        self.Wh2y = np.random.randn(self.output_dim, self.hidden_size2) * 0.01 # Weight matrix for hidden2 to output
        self.bh1 = np.zeros((self.hidden_size1, 1)) # hidden1 bias
        self.bh2 = np.zeros((self.hidden_size2, 1)) # hidden2 bias
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
        # print "Shape1", self.Wxh1.shape
        # print "Shape2", X.shape
        # print "Shape3", self.Wxh1.shape
        # print 'Shape4', self.Wh1h2.shape
        '''print "typeof X ", type(X)

        if (type(X)=="list"):
            print "X is ", X'''
        h_a1 = np.tanh(np.dot(self.Wxh1, np.reshape(X,(len(X),1))) + self.bh1)
        h_a2 = np.tanh(np.dot(self.Wh1h2, h_a1) + self.bh2)
        ys = np.exp(np.dot(self.Wh2y, h_a2) + self.by)
        probs = ys/np.sum(ys)
        return h_a1, h_a2, probs



    # def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
    #     """
    #     Add regularization terms to the weights
    #     :param dWhy: weight derivative from hidden to output
    #     :param dWxh: weight derivative from input to hidden
    #     :param Why: weights from hidden to output
    #     :param Wxh: weights from input to hidden
    #     :return: dWhy, dWxh
    #     """
    #     dWhy += self.reg_lambda * Why
    #     dWxh += self.reg_lambda * Wxh
    #     return dWhy, dWxh

    def _update_parameter(self, dWxh1, dWh1h2, dbh1, dbh2, dWh2y, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        self.Wxh1 += -self.learning_rate * dWxh1
        self.Wh1h2 += -self.learning_rate * dWh1h2
        self.bh1 += -self.learning_rate * dbh1
        self.bh2 += -self.learning_rate * dbh2
        self.Wh2y += -self.learning_rate * dWh2y
        self.by += -self.learning_rate * dby

    def _back_propagation(self, X, t, h_a1, h_a2, probs):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        dWxh1, dWh1h2, dWh2y = np.zeros_like(self.Wxh1), np.zeros_like(self.Wh1h2), np.zeros_like(self.Wh2y)
        dbh1, dbh2, dby = np.zeros_like(self.bh1), np.zeros_like(self.bh2), np.zeros_like(self.by)
        dy = np.copy(probs)
        #dy[t] -= 1
        # computing loss
        t = np.array(t)
        kd = probs.reshape(np.size(t))

        for i in range(len(kd)):
            dy[i]= kd[i]-t[i]


        dWh2y = np.dot(dy, h_a2.T)
        # print "dWh2y", dWh2y.shape
        # print "dby", dby.shape
        # print "dy", dy.shape
        dby += dy
        # pdb.set_trace()
        dh2 = np.dot(self.Wh2y.T, dy)  # backprop into h2
        dh1 = np.dot(self.Wh1h2.T, dh2)  # backprop into h1
        dh2raw = (1 - h_a2 * h_a2) * dh2 # backprop through tanh nonlinearity
        dbh2 += dh2raw
        dh1raw = (1 - h_a1 * h_a1) * dh1 # backprop through tanh nonlinearity
        # print "dbh2 ", dbh2.shape
        # print "dh2raw ", dh2raw.shape
        # print "dbh1 ", dbh1.shape
        # print "dh1raw ", dh1raw.shape
        # print "h_a1 ", h_a1.shape
        # print "dh1 ", dh1.shape
        # print "h_a2 ", h_a2.shape

        # print "Wh1h2 ", self.Wh1h2.T.shape
        # print "dWh1h2 ", dWh1h2.shape

        dbh1 += dh1raw

        # pdb.set_trace()
        dWh1h2 += np.dot(dh2raw, dh1raw.T)
        dWxh1 += np.dot(dh1raw, np.reshape(X, (len(X), 1)).T)
        return dWxh1, dWh1h2, dWh2y, dbh1, dbh2, dby

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
            # print 'ip len' ,len(inputs)
            # print 'tar len ', len(targets)
            for i in xrange(len(inputs)):
                # Forward pass
                h_a1, h_a2, probs = self._feed_forward(inputs[i])
                # print targets[i]
                loss += -np.log(probs[targets[i], 0])
                # print 'i ', i

                # Backpropagation
                dWxh1, dWh1h2, dWh2y, dbh1, dbh2, dby = self._back_propagation(inputs[i], targets[i], h_a1, h_a2, probs)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh1, dWh1h2, dbh1, dbh2, dWh2y, dby)

            # validation using the validation data

            validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            print 'Validation'

            for i in xrange(len(validation_inputs)):
                 # Forward pass
                h_a1, h_a2, probs = self._feed_forward(inputs[i])
                loss += -np.log(probs[targets[i], 0])
                # print 'llallslss' , loss
            #     # Backpropagation
                dWxh1, dWh1h2, dWh2y, dbh1, dbh2, dby = self._back_propagation(inputs[i], targets[i], h_a1,h_a2, probs)

            #     if regularizer_type == 'L2':
            #         dWhy, dWxh = self._regularize_weights(dWhy, dWxh, self.Why, self.Wxh)

            #     # Perform the parameter update with gradient descent
                self._update_parameter(dWxh1, dWh1h2, dbh1, dbh2, dWh2y, dby)

            if k%1 == 0:
                print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        h_a1, h_a2, probs = self._feed_forward(X)
        #return probs
        return np.argmax(probs)

    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

if __name__ == "__main__":
    nn = NeuralNetwork(768,16,8,5)
    data = []
    data = pickle.load(open('inputVectors.pkl','r'))
    autoencoder1 = pickle.load(open('autoencoder1.pkl','rb'))
    autoencoder2 = pickle.load(open('autoencoder2.pkl','rb'))
    nn.Wxh1 = autoencoder1.Wxh
    nn.Wh1h2 = autoencoder2.Wxh
    inputs = []
    targets=[]
    overall_list=[]
    for i in data['airplanes']:
        overall_list.append((i,[1,0,0,0,0]))
    for i in data['butterfly']:
        overall_list.append((i,[0,1,0,0,0]))
    for i in data['car_side']:
        overall_list.append((i,[0,0,1,0,0]))
    for i in data['Leopards']:
        overall_list.append((i,[0,0,0,1,0]))
    for i in data['Motorbikes']:
        overall_list.append((i,[0,0,0,0,1]))
    random.shuffle(overall_list)
    # print overall_list
    for i in overall_list:
        inputs.append(i[0])
        targets.append(i[1])
#     for i in range(1000):
#         num = random.randint(0,3)
#         inp = np.zeros((4,))
#         inp[num] = 1
#         inputs.append(inp)
#         targets.append(num)
    print targets
    count=0
    nn.train(inputs[:1000], targets[:1000], (inputs[1000:1200], targets[1000:1200]), 10, regularizer_type=None)
    for i in range(1000,1040):
        predicted = nn.predict(inputs[i])
        print predicted
        if (predicted!=np.argmax(np.array(targets[i]))):
            pass
        else:
            count+=1
    print count , " " , len(overall_list)-1000
    # print "percentage - ", (count/(len(overall_list)-1000) )* 100
    print "percentage - " , (count / 40) * 100
