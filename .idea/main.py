# Importing necessary libraries
import numpy as np
import gym
import random

# Initializing the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="human").env
# Resetting the environment to the starting state
env.reset()
# Displaying the initial environment state
env.render()

# Getting the number of possible actions in the environment
action_size = env.action_space.n
# Printing the number of possible actions
print("Action size ", action_size)

# Getting the number of possible states in the environment
state_size = env.observation_space.n
# Printing the number of possible states
print("State size ", state_size)

# Initializing the Q-table with zeros for all state-action pairs
qtable = np.zeros((state_size, action_size))
# Printing the initialized Q-table
print(qtable)

# Setting hyperparameters for the Q-learning algorithm
total_episodes = 50000        # Total episodes for training
total_test_episodes = 100     # Total episodes for testing
max_steps = 99                # Maximum steps allowed per episode
learning_rate = 0.7           # Learning rate for Q-value updates
gamma = 0.618                 # Discount factor for future rewards

# Parameters for exploration-exploitation trade-off
epsilon = 1.0                 # Initial exploration probability
max_epsilon = 1.0             # Maximum exploration probability
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Rate at which exploration probability decays

# Training loop
for episode in range(total_episodes):
    # Resetting the environment for a new episode
    state = env.reset()
    step = 0
    done = False

    # Loop for each step in the episode
    for step in range(max_steps):
        # Random number to decide between exploration and exploitation
        exp_exp_tradeoff = random.uniform(0,1)

        # If the number is greater than epsilon, we choose the action with the highest Q-value (exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        # Otherwise, choose a random action (exploration)
        else:
            action = env.action_space.sample()

        # Execute the chosen action and observe the new state and reward
        new_state, reward, done, info = env.step(action)

        # Update the Q-value for the current state-action pair
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        # Set the new state as the current state
        state = new_state

        # End the episode if the task is done
        if done == True:
            break

    # Decay the exploration probability
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

# List to store rewards for each testing episode
rewards = []

# Testing loop
for episode in range(total_test_episodes):
    # Resetting the environment for a new episode
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    # Printing the episode number
    print("****************************************************")
    print("EPISODE ", episode)

    # Loop for each step in the episode
    for step in range(max_steps):
        # Display the current environment state
        env.render()
        # Choosing the action with the highest Q-value for the current state
        action = np.argmax(qtable[state,:])

        # Execute the chosen action and observe the new state and reward
        new_state, reward, done, info = env.step(action)

        # Accumulate the rewards
        total_rewards += reward

        # End the episode if the task is done
        if done:
            rewards.append(total_rewards)
            # Printing the score for the episode
            print ("Score", total_rewards)
            break

        # Set the new state as the current state
        state = new_state

# Close the environment
env.close()
# Printing the average reward over all testing episodes
print ("Score over time: " +  str(sum(rewards)/total_test_episodes))
