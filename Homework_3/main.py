import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from models import BaseLearningAlgorithm, DeepNeuralNetworkLearningAlgorithm


#@title Download MNIST, split, and plot example
def plot_example(x_raw, y_raw):
  fig, axes = plt.subplots(3, 3)
  i = 0
  for i in range(3):
    for j in range(3):
      imgplot = axes[i,j].imshow(x_raw[i*3 + j].reshape((28,28)), cmap = 'bone')
      axes[i,j].set_title(y_raw[i*3 + j])
      axes[i,j].get_yaxis().set_visible(False)
      axes[i,j].get_xaxis().set_visible(False)
  fig.set_size_inches(18.5, 10.5, forward=True)


#@title Define a basic train and evaluation pipeline
def train_eval(learning_algo: BaseLearningAlgorithm, x_train, y_train,x_val, y_val, x_test, y_test):
  """Trains and evaluates the generic model."""
  y_test = y_test.astype('float32')
  learning_algo.fit(x_train, y_train, x_val, y_val)
  y_pred = learning_algo.predict(x_test)
  mat = confusion_matrix(y_test, y_pred)
  sns.set(rc = {'figure.figsize':(8,8)})
  sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
              xticklabels=['%d' %i for i in range(10)],
              yticklabels=['%d' %i for i in range(10)])
  plt.xlabel('true label')
  plt.ylabel('predicted label')
  plt.title(learning_algo.name)

  print(classification_report(y_test, y_pred,
                              target_names=['%d' %i for i in range(10)]))
  

def generate_mini_batches(X, Y, batch_size=64):
    m = X.shape[0]  # total number of samples
    mini_batches = []

    # Shuffle data
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    # Calculate the number of complete batches
    num_complete_batches = m // batch_size

    # Generate complete batches
    for k in range(0, num_complete_batches):
        batch_X = X_shuffled[k * batch_size: (k + 1) * batch_size]
        batch_Y = Y_shuffled[k * batch_size: (k + 1) * batch_size]
        mini_batches.append((batch_X, batch_Y))

    # Handle the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        batch_X = X_shuffled[num_complete_batches * batch_size:]
        batch_Y = Y_shuffled[num_complete_batches * batch_size:]
        mini_batches.append((batch_X, batch_Y))

    return mini_batches


if __name__ == '__main__':
  # Download the MNIST data set
  x_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True)
  x = (x_raw/255).astype('float32').to_numpy()
  # Split the data set into train, validation, and test sets.
  x_trainval, x_test, y_trainval, y_test = train_test_split(x, y_raw, test_size=0.10, random_state=42)
  x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.10, random_state=42)
  print('Here are the first 9 digits of the MNIST data with label.')
  plot_example(x, y_raw)
  # Define the learning algorithm
  learning_algo = DeepNeuralNetworkLearningAlgorithm(sizes=[784, 128, 64, 10], epochs=1, l_rate=0.001,
                                                    activation_function='sigmoid')
  # Train and evaluate the model
  st = time.monotonic()
  train_eval(learning_algo, x_train, y_train, x_val, y_val, x_test, y_test)
  et = time.monotonic()
  print(f'Training and evaluation took {(et - st):.2f} seconds')