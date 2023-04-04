# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:52:42 2023

@author: voche
"""
import numpy as np

#%% based on LMS
def adaptive_beamforming_LMS(signals, weights, stepsize=0.1):
  """
  Perform adaptive beamforming using the LMS algorithm.
  
  Parameters:
  signals: list of ndarray, the signals received by each antenna in the array
  weights: list of ndarray, the initial weights of each antenna
  stepsize: float, the step size for the LMS algorithm
  
  Returns:
  ndarray, the beamformed signal
  """
  num_antennas = len(signals)
  num_samples = signals[0].shape[0]
  
  # Initialize the beamformed signal to zero
  beamformed_signal = np.zeros(num_samples)
  
  # Iterate over each sample in the signals
  for i in range(num_samples):
    # Compute the output of the array by summing the weighted signals
    output = 0
    for j in range(num_antennas):
      output += weights[j] * signals[j][i]
      
    # Compute the error between the output and the desired signal
    error = signals[i] - output
    
    # Update the beamformed signal
    beamformed_signal[i] = output + error
    
    # Update the weights using the LMS algorithm
    for j in range(num_antennas):
      weights[j] += stepsize * error * signals[j][i]
      
  return beamformed_signal




#%%LMS algorithm
def lms_beamforming(x, d, num_iterations, step_size):
    '''
    x: The input signals at each time step, represented as a 2D numpy array with dimensions (number of sensors x number of time steps)
    d: The desired output signal at each time step, represented as a 1D numpy array with length equal to the number of time steps
    num_iterations: The number of iterations to run the LMS algorithm
    step_size: The step size, which controls the rate at which the weights are updated
    '''
    # Initialize weights
    weights = np.zeros(x.shape[0])
    
    for i in range(num_iterations):
        # Get the output of the beamformer
        y = np.dot(weights, x)
        
        # Calculate the error
        error = d[i] - y
        
        # Calculate the gradient
        grad = -error * x
        
        # Update the weights
        weights = weights - step_size * grad
    
    return weights


#%% RLS beamforming algorithm
def rls_beamforming(x, d, num_iterations, forgetting_factor, reg_factor):
    '''
    x: The input signals at each time step, represented as a 2D numpy array with dimensions (number of sensors x number of time steps)
    d: The desired output signal at each time step, represented as a 1D numpy array with length equal to the number of time steps
    num_iterations: The number of iterations to run the RLS algorithm
    forgetting_factor: The forgetting factor, which controls how much weight is given to past data when updating the inverse correlation matrix
    reg_factor: The regularization factor, which helps to stabilize the inverse correlation matrix and prevent it from becoming singular
    
    returns:
    weights: Final weigth of the beamformer
    '''
    # Initialize weights and inverse correlation matrix
    weights = np.zeros(x.shape[0])
    inv_corr = np.eye(x.shape[0]) / reg_factor
    
    for i in range(num_iterations):
        # Get the output of the beamformer
        y = np.dot(weights, x)
        
        # Calculate the error
        error = d[i] - y
        
        # Calculate the gradient
        grad = -error * x
        
        # Update the inverse correlation matrix
        inv_corr = (1 / forgetting_factor) * inv_corr + np.outer(x, x)
        
        # Calculate the step size
        step_size = np.dot(inv_corr, grad)
        
        # Update the weights
        weights = weights - step_size
    
    return weights

#%% SD beamforming algorithm
def steepest_descent_beamforming(x, d, num_iterations, step_size):
    '''
    x: The input signals at each time step, represented as a 2D numpy array with dimensions (number of sensors x number of time steps)
    d: The desired output signal at each time step, represented as a 1D numpy array with length equal to the number of time steps
    num_iterations: The number of iterations to run the steepest descent algorithm
    step_size: The step size, which controls the rate at which the weights are updated
    '''
    # Initialize weights
    weights = np.zeros(x.shape[0])
     
    for i in range(num_iterations):
        # Get the output of the beamformer
        y = np.dot(weights, x)
         
        # Calculate the error
        error = d[i] - y
         
        # Calculate the gradient
        grad = -error * x
         
        # Update the weights
        weights = weights - step_size * grad
         
    return weights

#%%CG algorithm
def conjugate_gradient_beamforming(x, d, num_iterations, step_size):
    '''
    x: The input signals at each time step, represented as a 2D numpy array with dimensions (number of sensors x number of time steps)
    d: The desired output signal at each time step, represented as a 1D numpy array with length equal to the number of time steps
    num_iterations: The number of iterations to run the conjugate gradient algorithm
    step_size: The step size, which controls the rate at which the weights are updated
    '''
    # Initialize weights and search direction
    weights = np.zeros(x.shape[0])
    search_dir = np.zeros(x.shape[0])
    
    for i in range(num_iterations):
        # Get the output of the beamformer
        y = np.dot(weights, x)
        
        # Calculate the error
        error = d[i] - y
        
        # Calculate the gradient
        grad = -error * x
        
        # Calculate the conjugate direction
        beta = np.dot(grad, grad) / np.dot(search_dir, grad)
        search_dir = grad + beta * search_dir
        
        # Update the weights
        weights = weights - step_size * search_dir
    
    return weights


#%%Kalman filter algo - method 1

# NOTE: To use this beamformer, create an instance of the KalmanFilterBeamformer 
# class and call the update method in a loop, passing in the measurements
class KalmanFilterBeamformer:
  def __init__(self, num_sensors, num_sources):
    self.num_sensors = num_sensors
    self.num_sources = num_sources
    self.A = np.eye(num_sources)  # state transition matrix
    self.C = np.zeros((num_sensors, num_sources))  # measurement matrix
    self.Q = np.eye(num_sources)  # process noise covariance
    self.R = np.eye(num_sensors)  # measurement noise covariance
    self.P = np.eye(num_sources)  # initial state covariance
    self.x_hat = np.zeros((num_sources, 1))  # initial state estimate
    self.x_hat_new = np.zeros((num_sources, 1))  # updated state estimate
    self.S = np.zeros((num_sensors, num_sensors))  # innovation covariance
    self.K = np.zeros((num_sources, num_sensors))  # Kalman gain

  def update(self, y):
    # Predict new state estimate using the state transition matrix
    self.x_hat_new = np.dot(self.A, self.x_hat)

    # Calculate the innovation covariance
    self.S = np.dot(self.C, np.dot(self.P, self.C.T)) + self.R

    # Calculate the Kalman gain
    self.K = np.dot(self.P, np.dot(self.C.T, np.linalg.inv(self.S)))

    # Update the state estimate using the Kalman gain and the measurement
    self.x_hat_new += np.dot(self.K, (y - np.dot(self.C, self.x_hat_new)))

    # Update the state covariance
    self.P = np.dot((np.eye(self.num_sources) - np.dot(self.K, self.C)), self.P)

    # Set the current state estimate to the updated state estimate
    self.x_hat = self.x_hat_new

  def get_weights(self):
    return self.x_hat

#%% kalman filter algo - method 2

class KalmanFilterAdaptiveBeamforming:
    def __init__(self, process_noise_covariance, measurement_noise_covariance):
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance
        self.state_vector = None
        self.covariance_matrix = None

    def initialize(self, initial_state_vector, initial_covariance_matrix):
        self.state_vector = initial_state_vector
        self.covariance_matrix = initial_covariance_matrix

    def predict(self, transition_matrix):
        self.state_vector = transition_matrix @ self.state_vector
        self.covariance_matrix = transition_matrix @ self.covariance_matrix @ transition_matrix.T + self.process_noise_covariance

    def update(self, measurement, measurement_matrix):
        residual = measurement - measurement_matrix @ self.state_vector
        Kalman_gain = self.covariance_matrix @ measurement_matrix.T @ np.linalg.inv(measurement_matrix @ self.covariance_matrix @ measurement_matrix.T + self.measurement_noise_covariance)
        self.state_vector = self.state_vector + Kalman_gain @ residual
        self.covariance_matrix = (np.eye(self.covariance_matrix.shape[0]) - Kalman_gain @ measurement_matrix) @ self.covariance_matrix

#%% Newtons' method

def NM_adaptive_beamforming(sensors, signals, max_iter=100, tol=1e-3):
    # Initialize the weight vector with all ones
    weights = np.ones(len(sensors))
    
    # Compute the steering vector
    steering_vector = compute_steering_vector(sensors)
    
    # Iterate until convergence or maximum number of iterations is reached
    for i in range(max_iter):
        # Compute the output signal
        output = compute_output_signal(weights, steering_vector, signals)
        
        # Compute the gradient
        gradient = compute_gradient(output, steering_vector, signals)
        
        # Compute the Hessian matrix
        hessian = compute_hessian(steering_vector, signals)
        
        # Update the weight vector using Newton's method
        weights -= np.linalg.solve(hessian, gradient)
        
        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            break
    
    return weights

def compute_steering_vector(sensors):
    # Assume that the sensors are equally spaced in a linear array
    # and that the signals are coming from a single source at an angle theta
    c = 3e8 # speed of light
    f = 2.4e9 # frequency
    d = 0.5 # inter-sensor distance
    theta = np.pi/4 # source angle
    
    steering_vector = np.exp(-1j * 2 * np.pi * f / c * d * np.cos(theta) * np.arange(len(sensors)))
    return steering_vector

def compute_output_signal(weights, steering_vector, signals):
    output = np.sum(weights * signals * steering_vector)
    return output

def compute_gradient(output, steering_vector, signals):
    gradient = 2 * np.conj(output) * (signals * steering_vector - output * weights)
    return gradient

def compute_hessian(steering_vector, signals):
    hessian = 2 * np.outer(steering_vector, np.conj(steering_vector))
    return hessian

#%% Supervised learning template
import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_shape)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.losses.MeanSquaredError(),
              metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

# Use the model to predict the beamforming weights for a new set of received signals
predictions = model.predict(X_new)

#%% Reinforcement learning- method 1
import gym
import numpy as np

# Create the environment
env = gym.make('AdaptiveBeamforming-v0')

# Set the discount factor and the learning rate
gamma = 0.9
alpha = 0.1

# Initialize the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Set the number of episodes to run
num_episodes = 1000

# Run the reinforcement learning loop
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()

    # Set the initial reward
    total_reward = 0

    # Run the episode loop
    while True:
        # Choose an action based on the current state and the Q-table
        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)*(1./(episode+1)))

        # Take the action and observe the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Update the Q-table
        Q[state, action] = Q[state, action] + alpha*(reward + gamma*np.max(Q[next_state,:]) - Q[state, action])

        # Increment the total reward
        total_reward += reward

        # Update the state
        state = next_state

        # Check if the episode is done
        if done:
            break

    # Print the total reward for the episode
    print("Episode {} finished with reward {}".format(episode+1, total_reward))

# Use the learned policy to select the beamforming weights
state = env.reset()
beamforming_weights = []
while True:
    action = np.argmax(Q[state,:])
    beamforming_weights.append(env.actions[action])
    state, reward, done, _ = env.step(action)
    if done:
        break

#%% Reinforcement learning- method 2
import tensorflow as tf
import numpy as np

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_shape)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.losses.MeanSquaredError())

# Set the discount factor and the learning rate
gamma = 0.9
alpha = 0.1

# Set the number of episodes to run
num_episodes = 1000

# Create a memory to store the transitions
memory = []

# Run the reinforcement learning loop
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()

    # Set the initial reward
    total_reward = 0

    # Run the episode loop
    while True:
        # Choose an action based on the current state and the model
        action = np.argmax(model.predict(state.reshape((1, -1))) + np.random.randn(1, env.action_space.n)*(1./(episode+1)))

        # Take the action and observe the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Store the transition in the memory
        memory.append((state, action, reward, next_state, done))

        # Increment the total reward
        total_reward += reward

        # Update the state
        state = next_state

        # Check if the episode is done
        if done:
            break

    # Print the total reward for the episode
    print("Episode {} finished with reward {}".format(episode+1, total_reward))

    # Sample a batch of transitions from the memory
    batch_size = 32
    batch = np.random.choice(len(memory), size=batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in batch])

    # Compute the Q-values for the next states
    Q_values = model.predict(np.vstack(next_states))
    max_Q_values = np.amax(Q_values, axis=1)

    # Compute the target Q-values for the states
    targets = rewards + gamma * max_Q_values * (1 - dones)

    # Update the model
    model.fit(np.vstack(states), targets, epochs=1, verbose=0)

# Use the learned policy to select the beamforming weights
state = env.reset()
beamforming_weights = []
while True:
    action = np.argmax(model.predict(state.reshape((1, -1))))
    beamforming_weights.append(env.actions[action])
    state, reward, done, _ = env.step(action)
    if done:
        break

#%% Reinforcement learning- method 3
import tensorflow as tf

class RL_AdaptiveBeamforming:
  def __init__(self, num_antennas, learning_rate=0.01):
    self.num_antennas = num_antennas
    self.learning_rate = learning_rate

    # Define the model
    self.model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_antennas, input_shape=(num_antennas,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate))

  def choose_antennas(self, antenna_gains):
    # Predict which antenna weights will result in the highest gain
    action = self.model.predict(antenna_gains)
    return action
    
  def update_weights(self, antenna_gains, target_gain, chosen_action):
    # Calculate the error between the predicted gain and the target gain
    error = target_gain - chosen_action
    
    # Use the error to update the weights of the model
    with tf.GradientTape() as tape:
      loss = self.model(antenna_gains, chosen_action)
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimize(zip(grads, self.model.trainable_variables), self.learning_rate)

#%% Unsupervised learning template
import tensorflow as tf

# Define the input data
input_data = tf.placeholder(tf.float32, shape=[None, num_input_features])

# Define the weights and biases for the beamforming filter
weights = tf.Variable(tf.random_normal([num_input_features, 1]))
biases = tf.Variable(tf.random_normal([1]))

# Define the beamforming filter
beamforming_filter = tf.matmul(input_data, weights) + biases

# Define the loss function
loss = tf.reduce_mean(tf.square(beamforming_filter - target_output))

# Define the optimization algorithm
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Train the model
with tf.Session() as sess:
  sess.run(init)
  for i in range(num_steps):
    sess.run(optimizer, feed_dict={input_data: input_data_batch, target_output: target_output_batch})


