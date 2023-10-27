#@title Imports
import time
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from keras.utils import to_categorical
import tensorflow as tf
from main import generate_mini_batches


#@title Define a base class class for Learning Algorithm
class BaseLearningAlgorithm(ABC):
  """Base class for a Supervised Learning Algorithm."""
  @abstractmethod
  def fit(self, x_train:np.array, y_train: np.array
          , x_val:np.array
          , y_val:np.array) -> None:
    """Trains a model from labels y and examples X.
    We include validation set for optional hyperparameter tuning. Not
    all of the algorithims we use will
    """

  @abstractmethod
  def predict(self, x_test: np.array) -> np.array:
    """Predicts on an unlabeled sample, X."""

  @property
  @abstractmethod
  def name(self) -> str:
    """Returns the name of the algorithm."""


#@title Define the basic Logistic Regression Model
class LogisticRegressionLearningAlgorithm(BaseLearningAlgorithm):
  """Minimalist wrapper class for basic Logistic Regression."""
  def __init__(self, max_iters: int = 1000):
    self._model = LogisticRegression(max_iter = max_iters, verbose = True, penalty = 'l2')

  def fit(self, x_train:np.array, y_train: np.array, x_val:np.array, y_val: np.array) -> None:
    self._model.fit(x_train,y_train)
    # Don't need the validation data in Logistic Regression

  def predict(self, x_test: np.array) -> np.array:
    return self._model.predict(x_test)

  @property
  def name(self) -> str:
    return 'Logistic Regression'


class RadialBasisSvmLearningAlgorithm(BaseLearningAlgorithm):
  """RBF SVM Classifier function."""
  def __init__(self, cost: float, gamma: float):
    self.cost = cost
    self.gamma = gamma
    self._model = SVC(C=cost, gamma=gamma, kernel='rbf')

  def fit(self, x_train:np.array, y_train: np.array, x_val:np.array, y_val: np.array) -> None:
    self._model.fit(x_train, y_train)

  def predict(self, x_test: np.array) -> np.array:
    return self._model.predict(x_test)

  @property
  def name(self) -> str:
    return 'Radial Basis SVM'
  

class RadialBasisSvmPcaLearningAlgorithm(RadialBasisSvmLearningAlgorithm):
  """RBM SVM Classifier with PCA applied class"""
  def __init__(self, cost: float, gamma: float, num_pca_components: int):
    super().__init__(cost, gamma)
    self.num_pca_components = num_pca_components
    self._model = make_pipeline(
        PCA(n_components=self.num_pca_components),
        SVC(C=self.cost, gamma=self.gamma, kernel='rbf')
    )

def fit(self, x_train:np.array, y_train: np.array, x_val:np.array, y_val: np.array) -> None:
  self._model.fit(x_train, y_train)

def predict(self, x_test: np.array) -> np.array:
  return self._model.predict(x_test)

@property
def name(self) -> str:
  return "Radial Basis SVM with PCA"


class RandomForestLearningAlgorithm(BaseLearningAlgorithm):
  """Random Forest Classifier class"""
  def __init__(self, n_estimators: int, criterion: str, max_depth: int, min_samples_split: int):
    self.n_estimators = n_estimators
    self.criterion = criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self._model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

  def fit(self, x_train:np.array, y_train: np.array, x_val:np.array, y_val: np.array) -> None:
    self._model.fit(x_train, y_train)

  def predict(self, x_test: np.array) -> np.array:
    return self._model.predict(x_test)

  @property
  def name(self) -> str:
    return "Random Forest Classifier"


#@title Basic Neural Network Algorithm

class DeepNeuralNetworkLearningAlgorithm(BaseLearningAlgorithm):
    def __init__(self, sizes, epochs=10, l_rate=0.001, batch_size=64, activation_function='relu', early_stopping=False, patience=5, min_delta=0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        # choose the supported activation function
        if activation_function == 'sigmoid':
          self.activation = self.sigmoid
        elif activation_function == 'relu':
          self.activation = self.relu
        else:
          raise ValueError(f"Unsupported activation function: {activation_function}")

        # Save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def relu(self, x, derivative=False):
        if derivative:
            return (x > 0).astype(int)
        return np.maximum(0, x)

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # He initialization fore ReLU
        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        # Xavier initialization for sigmoid (does not work well for some reason)
        # params = {
        #     'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / input_layer),
        #     'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_1),
        #     'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / hidden_2)
        # }

        return params

    def compute_average_gradient(self, batch_gradients):
        average_gradient = {}
        for key in batch_gradients[0].keys():
            average_gradient[key] = np.mean([grad[key] for grad in batch_gradients], axis=0)
        return average_gradient

    def forward_pass(self, x_train: np.array) -> np.array:
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.activation(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.activation(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train: np.array, output: np.array) -> Dict[str, np.array]:
        """Perfoms backpropagation."""
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.activation(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.activation(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w: Dict[str, np.array]):
        """Updates network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        """

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        """
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        """
        predictions = []
        y_val = to_categorical(y_val)
        for x, y in zip(x_val, y_val):

            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def predict(self, X_test: np.array) -> np.array:
      """Predicts on a test set."""
      return np.array([str(np.argmax(self.forward_pass(x))) for x in X_test]).astype('float32')

    def fit(self, x_train, y_train, x_val, y_val):
        best_val_accuracy = 0
        best_params = None
        epochs_without_improvement = 0

        y_train = to_categorical(y_train)

        # mini-batch
        start_time = time.time()
        for iteration in range(self.epochs):
            mini_batches = generate_mini_batches(x_train, y_train, self.batch_size)
            for (x_batch, y_batch) in mini_batches:
                batch_gradients = []
                for x, y in zip(x_batch, y_batch):
                    output = self.forward_pass(x)
                    changes_to_w = self.backward_pass(y, output)
                    batch_gradients.append(changes_to_w)
                average_gradient = self.compute_average_gradient(batch_gradients)
                self.update_network_parameters(average_gradient)

            accuracy = self.compute_accuracy(x_val, y_val)

            # early stopping is applied here
            if self.early_stopping:
              if accuracy - best_val_accuracy > self.min_delta:
                best_val_accuracy = accuracy
                best_params = self.params
                epochs_without_improvement = 0
              else:
                epochs_without_improvement += 1

              # loading best params
              if epochs_without_improvement > self.patience:
                print("Early stopping applied")
                self.params = best_params
                break

            print('Epoch: {0}, Time Spent: {1:.2f}s, Validation Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
    @property
    def name(self) -> str:
      return 'Basic Neural Network'
    

class KerasDnnLearningAlgorithm(BaseLearningAlgorithm):
  """Keras DNN Learning Algorithm"""
  def __init__(self, input_dim, output_dim, hidden_layers, activation, batch_size=32, epochs=10, learning_rate=0.001, patience=5, dropout=0.2):
    self.hidden_layers = hidden_layers
    self.activation = activation
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.patiance = patience
    self.dropout = dropout
    self._model = self._get_model(input_dim, output_dim, hidden_layers, activation)

  def _get_model(self, input_dim, output_dim, hidden_layers, activation):
    model = tf.keras.Sequential()
    
    # input layer
    model.add(
      tf.keras.layers.Dense(
        input_dim, input_dim=input_dim, activation=activation))

    # hidden and droput layers
    for layer_width in hidden_layers:
      model.add(
        tf.keras.layers.Dense(
          layer_width, activation=activation))
      model.add(tf.keras.layers.Dropout(self.dropout))

    # output layer
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

    model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
      metrics=['accuracy'])
    return model

  def predict(self, X_test: np.array) -> np.array:
    """Predicts on a test set."""
    # return class labels by taking the max element index using the last dimension
    return np.argmax(self._model.predict(X_test), axis=-1)

  def fit(self, x_train, y_train, x_val, y_val):
    y_train = y_train.astype('int32')
    y_val = y_val.astype('int32')
    early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', patience=self.patiance)
    self._model.fit(
       x_train, y_train,
       epochs=self.epochs,
       validation_data=(x_val, y_val),
       callbacks=[early_stopping],
       batch_size=self.batch_size
    )

  @property
  def name(self) -> str:
     return 'Keras DNN'
