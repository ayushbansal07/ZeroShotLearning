from __future__ import print_function
import numpy as np
import math

class RBM:

  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True


    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):


    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):

      pos_hidden_activations = np.dot(data, self.weights)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

      pos_associations = np.dot(data.T, pos_hidden_probs)

      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)

      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum(np.square(data - neg_visible_probs))
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):



    num_examples = data.shape[0]

    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)


    hidden_activations = np.dot(data, self.weights)

    hidden_probs = self._logistic(hidden_activations)

    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)


    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):


    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

    visible_states = visible_states[:,1:]
    return visible_states

  def daydream(self, num_samples):

    samples = np.ones((num_samples, self.num_visible + 1))


    samples[0,1:] = np.random.rand(self.num_visible)


    for i in range(1, num_samples):
      visible = samples[i-1,:]

      hidden_activations = np.dot(visible, self.weights)

      hidden_probs = self._logistic(hidden_activations)

      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      hidden_states[0] = 1
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    return samples[:,1:]

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

def train_RBM_and_compute_simiarity(class_array,target_filename,reduced_dimension=100,epochs=5000):
  #num_tags = 1000    #number of tags
  #reduced_demension = 100 #reduced demension
  #epochs = 5000
  #filename =""    # name of np array containing one hot encoding for classes
  num_tags = class_array.shape[1]
  r = RBM(num_visible = num_tags, num_hidden = reduced_dimension)
  training_data = class_array
  r.train(training_data, max_epochs = epochs)
  print(r.weights)
  classes_rep = np.zeros((num_tags,num_tags),dtype=int)
  low_dem_rep = []
  for i in range(0,num_tags):
    classes_rep[i][i] = 1
    low_dem_rep.append(r.run_visible(classes_rep[i].reshape(1,num_tags)))
  simi_matrix = np.zeros((num_tags,num_tags))
  low_dem_rep =np.asarray(low_dem_rep)
  for i in range(0,num_tags):
    for j in range(i,num_tags):
      if (i == j):
        simi_matrix[i][j] = 1
      else :
        val  = math.exp(np.linalg.norm(low_dem_rep[i]-low_dem_rep[j])**2/10**2)
        simi_matrix[i][j] =val
        simi_matrix[j][i] = val
  np.save(target_filename,simi_matrix)
  return simi_matrix
