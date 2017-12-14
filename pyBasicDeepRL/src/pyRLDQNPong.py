'''
Created on Nov 28, 2017

@author: Francisco

From DeepLearningNanoDegree in Udacity
'''
import gym
import tensorflow as tf
import numpy as np
import cv2
from pyMemory import Memory
from pyWorld import World


# Create the Cart-Pole game environment
world=World()
rewards = []
for _ in range(1):
    action=world.actionSample()
    state, reward, done, info = world.step(action) # take a random action
    rewards.append(reward)
    if done:
        rewards = []
        world.reset()
print(rewards[-20:])


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=80*80, 
                 action_size=3, hidden_size=10, 
                 name='QNetwork'):
        with tf.variable_scope(name):
            # state inputs to the Q-network
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            # in Q-learning this target state action values is
            # Rs'a+gamma*max_a[Q(s',a)]
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 3, for three actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            # we work out the state action value Q(s,a) 
            self.Qsa = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Qsa))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    def getQ(self,state):
        feed = {self.inputs_: state.reshape((1, *state.shape))}
        Q = sess.run(self.output, feed_dict=feed)
        return Q
    def getQs(self,states):
        Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: states})
        return Qs
    def getBestAction(self,state):
            feed = {self.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(self.output, feed_dict=feed)
            action = np.argmax(Qs)
            return action

            

train_episodes = 10000        # max number of episodes to learn from
max_steps = 2000                # max steps in an episode
gamma = 0.999                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.05            # minimum exploration probability 
decay_rate = 0.000001          # exponential decay rate for exploration prob

# Network parameters
hidden_size = 100              # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20               # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)
# Initialize the simulation
world.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = world.stepSample()

memory = Memory(max_size=memory_size)

restore=True
 
# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = world.actionSample()
    next_state, reward, done, _ = world.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        
        # Start new episode
        world.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = world.stepSample()
        t=max_steps
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

# Now train with experiences
saver = tf.train.Saver()
rewards_list = []
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    if restore:
        print("Model restored")
        #saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        saver.restore(sess, "checkpoints/pong.ckpt")    
    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            # env.render() 
            
            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
            if explore_p > np.random.rand():
                # Make a random action
                action = world.actionSample()
            else:
                # Get action from Q-network
                action = mainQN.getBestAction(state)
            
            # Take action, get new state and reward
            next_state, reward, done, _ = world.step(action)
    
            total_reward += reward
            
            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)

                t = max_steps
                
                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))
                
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                
                # Start new episode
                world.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = world.stepSample()
            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1
            
            # Sample mini-batch from memory
            batch   = memory.sample(batch_size)
            states  = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            
            # Train network
            # given (s,a,r,s') or (s0,a,r,s1)
            # ask the network for Q(s',a or Q(s1,a)
            Qs1a = mainQN.getQs(next_states)
            #Qs1a = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
            
            # Set target_Qs to 0 for states where episode ends
            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            Qs1a[episode_ends] = (0, 0, 0)
            
            # The target of the NN is
            # R(s,a)+gamma*max_a[Q(s1,a)]
            targets = rewards + gamma * np.max(Qs1a, axis=1)

            loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                feed_dict={mainQN.inputs_: states,
                                           mainQN.targetQs_: targets,
                                           mainQN.actions_: actions})
        
            if ep%100==0:
                saver.save(sess, "checkpoints/pong.ckpt")
    
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
# eps, rews = np.array(rewards_list).T
# smoothed_rews = running_mean(rews, 10)
# plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
# plt.plot(eps, rews, color='grey', alpha=0.3)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')

test_episodes = 10
test_max_steps = 4000
world.reset()
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/pong.ckpt")
    
    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            world.render=True 
            
            # Get action from Q-network
            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)
            
            # Take action, get new state and reward
            next_state, reward, done, _ = world.step(action)
            
            if done:
                t = test_max_steps
                world.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = world.stepSample()
            else:
                state = next_state
                t += 1
                
world.env.close()