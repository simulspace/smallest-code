"""
Created on Mon Apr 30 00:09:59 2018

@author: avi_0
"""

#lets make full fledged neural network

from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        random.seed(1)#seeding the rNDIM numbr generator ,for generating same number everytime
        
        self.synaptic_weights = 2*np.random.random(3,1) - 1
        #here we created 3 input and 1 output connection neuron,with weights belonging from -1 to +1
        
        #now lets creat a sigmoid function wich takes passes weighted sum of input throught it
        #so by sigmoid we normalise it between 0,1
        def __sigmoid(self, x):
            return 1/(1 + exp(-x))
        
        #now take derrivitive of the sigmoid function so that it will tell how confident are we about the sigmoid function
        def __sigmoid_derrivative(self,x):
            return x*(1-x)
        
        #now time to train the network with our data
        
        def train(self, training_set_input, training_set_output, number_of_training_iterations):
            for iterations in range(number_of_training_iterations):
                output= self.think(training_set_inputs)
        
                error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print ("Considering new situation [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))
