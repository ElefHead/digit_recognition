from filereader import Filereader
import numpy as np 
np.random.seed(1)


class NeuralNet :

    def __init__(self, layers=[200, 200, 10], activation=['elu','elu','softmax'],
                 epochs=100, elu_alpha=1.2, batch_size=250):
        self.learning_rate = 0.001
        self.epochs = epochs
        self.num_layers = len(layers)
        self.layers = layers
        self.activation = activation
        self.elu_alpha = elu_alpha
        self.activate = {
            'elu': self.elu_activation,
            'softmax': self.softmax_activation
        }
        self.weights = []
        self.bias = []
        self.batch_size = batch_size
        self.differentiate = {
            'elu': self.d_elu_activation,
            'softmax': self.d_softmax_activation
        }
        self.momentum_cache = {}

    def forward_pass(self, train, save_cache=False):
        cache = {
            'scores': [],
            'inputs': [],
            'past_weights': [],
            'past_bias':[]
        }
        for i,n in enumerate(self.layers):
            if i == 0:
                Z = np.dot(train,self.weights[i]) + self.bias[i]
            else:
                Z = np.dot(A,self.weights[i]) + self.bias[i]
            if save_cache:
                cache['scores'].append(Z)
                if i!=0:
                    cache['inputs'].append(A)
            A = self.activate[self.activation[i]](Z)
        return (A, cache) if save_cache else (A, None)

    def backpropogate_update(self, X_train, Y_train, prediction, cache):
        batch_size = X_train.shape[0]
        d_output = self.d_categorical_cross_entropy_loss(Y_train,prediction)
        if 'past_weights' not in self.momentum_cache:
            self.momentum_cache['past_weights'] = [np.zeros_like(w) for w in self.weights]
            self.momentum_cache['past_bias'] = [np.zeros_like(b) for b in self.bias]

        for layer in range(len(self.layers)-1,0,-1):
            d_score = d_output*self.differentiate[self.activation[layer]](cache['scores'][layer])
            if layer==0:
                d_weights = np.dot(d_score, X_train.T)/batch_size
            else:
                d_weights = np.dot(cache['inputs'][layer-1].T, d_score)/batch_size + 0.9*self.momentum_cache['past_weights'][layer]
            d_bias = np.sum(d_score, axis=0, keepdims=True) + 0.9*self.momentum_cache['past_bias'][layer]
            d_output = np.dot(d_score,self.weights[layer].T)
            self.weights[layer] -= (self.learning_rate * d_weights)
            self.bias[layer] -= (self.learning_rate * d_bias)
            # Just Momentum
            self.momentum_cache['past_weights'][layer] = d_weights
            self.momentum_cache['past_bias'][layer] = d_bias

    def softmax_activation(self, Z):
        Z_dash = Z - Z.max()  # for numerical stability
        e = np.exp(Z_dash)
        return e / (np.sum(e, axis=1, keepdims=True))

    def d_softmax_activation(self, y):
        return y * (1 - y)

    def elu_activation(self, Z):
        return np.where(Z >= 0, Z, self.elu_alpha*(np.exp(Z) - 1))

    def d_elu_activation(self, Z):
        return (Z >= 0).astype('float32') + (Z < 0).astype('float32') * (self.elu_activation(Z) + self.elu_alpha)

    def categorical_cross_entropy_loss(self, actual, prediction):
        prediction /= np.sum(prediction, axis=-1, keepdims=True)
        prediction = np.clip(prediction, 10e-8, 1. - 10e-8)  # for numerical stability
        return -np.sum(actual * np.log(prediction))

    def d_categorical_cross_entropy_loss(self, actual, prediction):
        return actual - prediction

    def init_weights(self,M):
        # using He normal initialization
        weights = []
        bias = []
        for n,m in enumerate(self.layers):
            if n==0:
                weights.append(np.random.normal(0, np.sqrt(2/M),size=[M,m]))
            else:
                weights.append(np.random.normal(0,np.sqrt(2/self.layers[n-1]),size=[self.layers[n-1],m]))
            bias.append(np.random.uniform(-0.2,0.2,size=[1,m]))
        return weights, bias

    def get_batch(self, X_train, Y_train):
        n_batches = X_train.shape[0]//self.batch_size
        if n_batches == 0:
            yield X_train, Y_train
        for i in range(n_batches):
            if i==n_batches-1:
                yield X_train[i*self.batch_size:, :], Y_train[i*self.batch_size:, :]
            else:
                yield X_train[i*self.batch_size:(i+1)*self.batch_size, :], Y_train[i*self.batch_size:(i+1)*self.batch_size, :]

    def train(self, X_train, Y_train):
        N, M = X_train.shape
        self.weights, self.bias = self.init_weights(M)

        shuffle_indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[shuffle_indices]
        Y_train_shuffled = Y_train[shuffle_indices]

        for i in range(self.epochs):
            for X_batch, Y_batch in self.get_batch(X_train_shuffled, Y_train_shuffled):
                prediction, cache = self.forward_pass(X_batch, save_cache=True)

                self.backpropogate_update(X_batch, Y_batch, prediction, cache)

            print("epoch {}: Training accuracy = {}".format(i+1, accuracy(self.predict(X_train), Y_train)))

    def predict(self,X):
        n_batches = X.shape[0] // self.batch_size
        output_size = self.layers[len(self.layers)-1]
        if n_batches == 0:
            predictions,cache = self.forward_pass(X,save_cache=False)
        else:
            predictions = np.zeros([X.shape[0],output_size])
            for i in range(n_batches):
                if i==n_batches-1:
                    predictions[i*self.batch_size:], cache = self.forward_pass(X[i*self.batch_size:])
                else:
                    predictions[i * self.batch_size:(i+1)*self.batch_size], \
                    cache = self.forward_pass(X[i * self.batch_size: (i+1)*self.batch_size])
        return predictions


def accuracy(actual, prediction):
    return np.mean(np.argmax(actual,axis=1)==np.argmax(prediction,axis=1))


def main():
    f = Filereader(path="./data/")
    X_train, Y_train, train_rows, train_cols = f.getData(sample=60000)							# init training data
    X_test, Y_test, test_rows, test_cols = f.getData(dataset="testing", sample=10000)			    # init testing data
    X_train = X_train/255
    X_test = X_test/255
    print("training data shape: {}".format(X_train.shape))
    print("training labels shape: {}".format(Y_train.shape))

    nn = NeuralNet()
    nn.train(X_train, Y_train)
    test_pred = nn.predict(X_test)

    print("Final Testing Accuracy = {}".format(accuracy(Y_test, test_pred)))


if __name__ == '__main__':
    main()

