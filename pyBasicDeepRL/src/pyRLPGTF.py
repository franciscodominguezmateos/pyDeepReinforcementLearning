'''
Created on Nov 20, 2017

@author: Francisco Dominguez
from:
https://www.youtube.com/watch?v=aRKOJHRbXeo

'''
import tensorflow as tf
import gym
import numpy as np
import cv2
from pyWorld import World

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def normalize_rewards(episode_rewards):
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    normalized_rewards = episode_rewards-np.mean(episode_rewards)
    normalized_rewards /= np.std(normalized_rewards)
    return normalized_rewards

class PGNetwork:
    def __init__(self):
        tf.reset_default_graph()
        self.observations=tf.placeholder(shape=[None,80*80],dtype=tf.float32)#pixels
        self.actions     =tf.placeholder(shape=[None],dtype=tf.uint8)#0,1,2 for up, still, down
        self.rewards     =tf.placeholder(shape=[None],dtype=tf.float32) #+1,-1, with discounts
        
        #model
        self.Y=tf.layers.dense(self.observations,200,activation=tf.nn.relu)
        self.Ylogits=tf.layers.dense(self.Y,3)
        
        #sample an action from predicted probabilities
        self.sample_op=tf.multinomial(logits=tf.reshape(self.Ylogits,shape=(1,3)),num_samples=1)
        
        #loss
        self.cross_entropies=tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.actions,3),logits=self.Ylogits)
        
        #loss=tf.reduce_sum(rewards*cross_entropies)
        self.loss=tf.reduce_mean(self.rewards*self.cross_entropies)
        
        #training operation
        self.optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.99)
        self.train_op=self.optimizer.minimize(self.loss)
        
    def getAction(self,state):
        action=sess.run(self.sample_op,feed_dict={self.observations:[state]})
        return action
    
    def train(self,npstates_batch,npactions_batch,nprewards_batch):
        feed_dict={self.observations:npstates_batch, self.actions:npactions_batch,self.rewards:nprewards_batch}
        sess.run(self.train_op,feed_dict=feed_dict)

def playEpisode(pgnn):
    #reset everything
    state=world.reset() # This gets us the image
    statesList =[]
    actionsList=[]
    rewardsList=[]
    done=False
    episode_reward=0
    while not done:
        #cv2.imshow("state",(state.reshape([80,80])+1)/2)
        #cv2.waitKey(-1)
        #decide what move to play: UP, STILL, DOWN (through NN model)
        action=pgnn.getAction(state)
        #play it (through openAI gym pong simulator)
        game_state, reward, done, info = world.step(action)
        #collect results
        statesList.append(state)
        actionsList.append(action)
        rewardsList.append(reward)
        episode_reward+=reward
    #process the rewards after each episode
    #Discountand Normalize reguards
    processed_rewards=discount_rewards (rewardsList,gamma)
    processed_rewards=normalize_rewards(processed_rewards)
    #Convert list to np.array
    npstates =np.vstack(statesList)
    npactions=np.vstack(actionsList)#[:,0]
    nprewards=np.vstack(processed_rewards)#[:,0]
    return npstates,npactions,nprewards,episode_reward

world = World()

pgnn=PGNetwork()

init = tf.initialize_all_variables()

render=False
restore=False
gamma = 0.99 # discount factor for reward
BATCH_SIZE=10
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    if restore: 
        print("Model restored")
        saver.restore(sess, "model.ckpt")
    num_episodes=0
    num_batches=0
    episode_reward_mean=-21
    while True:
        num_episodes_batch=0
        npstates_batch =np.empty((0,80*80))
        npactions_batch=np.empty((0,1))
        nprewards_batch=np.empty((0,1))
        while num_episodes_batch< BATCH_SIZE:
            #Play a episode
            npstates,npactions,nprewards,episode_reward=playEpisode(pgnn)
            #append data to the batch
            npstates_batch =np.vstack((npstates_batch ,npstates ))
            npactions_batch=np.vstack((npactions_batch,npactions))
            nprewards_batch=np.vstack((nprewards_batch,nprewards))
            #running mean of rewards
            episode_reward_mean=0.99*episode_reward_mean+0.01*episode_reward
            #counts
            num_episodes+=1
            num_episodes_batch+=1
            print(num_episodes_batch,"batch=",num_batches,"epmn=",episode_reward_mean,"eprw=",episode_reward)

        print("Training this batch with shape=",npstates_batch.shape)
        pgnn.train(npstates_batch,npactions_batch[:,0],nprewards_batch[:,0])
        num_batches+=1
        if num_batches%10==0:
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)

 
        
        
        
        
        
        
        