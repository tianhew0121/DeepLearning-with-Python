import numpy as np
import h5py

class NeuralNetwork(object):
    def __init__(self):
        self.weights = []
        self.activations = []
        self.biases = []
        self.grad_weights = []
        self.grad_biases = []

    def cross_entropy_loss(self, predicted, target):
        """
        predicted: NxC shape numpy array where predicted[:,i] in range [0; 1]
        target: NxC shape numpy array where target[:,i] = 1 if i is the target class else 0
        """
        n_samples = target.shape[0]
        loss = -np.sum(target * np.log2(predicted + 1e-8))/ n_samples
        return loss


    def _Relu(self, si):
        s = np.copy(si)
        s[s < 0] = 0
        return s
    
    def _Relu_derivative(self, si):
        return (si > 0) * 1

    def _tanh(self, s):
        return np.tanh(s)
    
    def _tanh_derivative(self, s):
        return 1 - np.power(np.tanh(s),2)
    
    def _softMax(self,x):
        exp_x = np.exp(x)
        exp_sum = np.sum(exp_x, axis=1)
        return exp_x / exp_sum[:,None]
    

    def add_layer(self, input_size, output_size, activation='relu'):
        """
        add a layer to the network
        This function will initialize a weight matrix of size MxN
        M = output size, N = input size
        activation: specifies the output of this layer
        At first, all the values in the matrix will be initialized randomly
        """
        if activation == 'relu':
            self.activations.append(self._Relu)
        if activation == 'tanh':
            self.activations.append(self._tanh)
        if activation == 'softmax':
            self.activations.append(self._softMax)
        self.weights.append(np.random.randn(input_size, output_size)/100)
        self.biases.append(np.random.randn(1, output_size))
        return self
    
    def forward(self, x):
        """
        x: NxD shape numpy array
        N: number of points
        D: number of dimension
        """
        N, D = x.shape
        self.input = x
        self.a = []
        self.s = []
        for layer_ind in range(len(self.weights)):
            x = x @ self.weights[layer_ind] + self.biases[layer_ind]
            act = self.activations[layer_ind](x)
            self.s.append(x)
            self.a.append(act)
        return act
    
    def backward(self, activation, y, learning_rate=0.01):
        """
        x: NxD shape numpy array
        y: N shape numpy array
        """
        N = y.shape[0]
        n_layer = len(self.weights)

        self.old_weights = [np.copy(self.weights[i]) for i in range(n_layer)]
        for i in range(n_layer):
            curr_index = n_layer-i-1
            if i == 0:
                dCds = (activation - y) / N
                self.weights[curr_index] -= learning_rate * (self.a[curr_index-1].T @ (dCds))
                self.biases[curr_index] -= learning_rate * (np.sum(dCds, axis=0))
            else:
                if self.activations[curr_index] == self._Relu:
                    relu_val = self._Relu_derivative(self.s[curr_index])
                    dads = relu_val
                if self.activations[curr_index] == self._tanh:
                    dads = np.sum(self._tanh_derivative(self.s[curr_index]), axis=0)
                dCds = np.multiply(dCds @ self.old_weights[curr_index + 1].T, dads)
                if curr_index == 0:
                    update = self.input.T @ dCds
                else:
                    update = self.a[curr_index-1].T @ dCds
                self.weights[curr_index] -= learning_rate * update 
                self.biases[curr_index] -= learning_rate * (np.sum(dCds, axis=0))
        return self
        

    def fit(self, X, y, vX, vy, epochs=100, batch_size=1000, learning_rate=0.01):
        """
        X: NxD shape numpy array
        y: N shape numpy array
        """
        N = X.shape[0]
        N_validate = vX.shape[0]
        print(N)
        epoch_acc_train = []
        epoch_acc_valid = []
        for epoch in range(epochs+1):
            for batch in range(0, N, batch_size):
                X_batch = X[batch:batch+batch_size]
                y_batch = y[batch:batch+batch_size]
                activation = self.forward(X_batch)
                self.backward(activation, y_batch, learning_rate)
                
            activation = self.forward(vX)
            epoch_acc_validation = np.round(np.sum(np.argmax(activation, axis=1) == np.argmax(vy, axis=1))/N_validate * 100, 2)
            activation = self.forward(X)
            epoch_acc_training = np.round(np.sum(np.argmax(activation, axis=1) == np.argmax(y, axis=1))/N * 100, 2)
            # epoch_loss_lst.append(epoch_loss)
            epoch_acc_valid.append(epoch_acc_validation)
            epoch_acc_train.append(epoch_acc_training)
            epoch_loss = np.round(self.cross_entropy_loss(activation, y), 4)

            if epoch % 10 == 0: 
                print(f"Epoch: {epoch} Loss = {epoch_loss} Accuracy train = {epoch_acc_training}% Accuracy validation = {epoch_acc_validation}%")
            if epoch % 20 == 0 and epoch != 0:
                learning_rate /= 2
        print("training finished")
        return epoch_acc_train, epoch_acc_valid
        


    

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    
    train_data_path = "data/mnist_traindata.hdf5"
    test_data_path = "data/mnist_testdata.hdf5"
    with h5py.File(train_data_path, 'r') as hf:
        X = hf['xdata'][:]
        y = hf['ydata'][:]
        train_target = np.argmax(y, axis=1)
    train_x, vX, train_y, vy = train_test_split(X, y, test_size=1/6, random_state=0)

    model = NeuralNetwork()
    model.add_layer(784, 128, 'tanh')
    model.add_layer(128, 128, 'tanh')
    model.add_layer(128, 10, 'softmax')
    loss_lst, acc_lst = model.fit(train_x, train_y, vX, vy, epochs=50, batch_size=500, learning_rate=0.01)
