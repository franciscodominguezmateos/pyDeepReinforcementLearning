'''
Created on Nov 20, 2017

@author: Francisco Dominguez
from:
https://www.youtube.com/watch?v=aRKOJHRbXeo
https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

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
        self.actions     =tf.placeholder(shape=[None],dtype=tf.int32)#0,1,2 for up, still, down
        self.rewards     =tf.placeholder(shape=[None],dtype=tf.float32) #+1,-1, with discounts
        
        #model
        self.h1=tf.layers.dense(self.observations,200,activation=tf.nn.relu)
        self.logits=tf.layers.dense(self.h1,3)
        
        #get probabilities
        self.prob=tf.nn.softmax(self.logits)
        
        #get log probabilities
        self.logprob=tf.log(self.prob+1e-13)
        
        #sample an action from predicted probabilities
        self.sample_op=tf.multinomial(logits=tf.reshape(self.logits,shape=(1,3)),num_samples=1)
        
        #get the best action, that with biggest logit
        self.best_action=tf.argmax(self.logits,axis=1)
        
        #onehot action
        self.one_hot=tf.one_hot(self.actions,3)
        
        #loss
        #self.cross_entropies=tf.losses.softmax_cross_entropy(onehot_labels=self.one_hot,logits=self.logits)
        self.cross_entropies=tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot,logits=self.logits)
        #self.cross_entropies=-tf.reduce_sum(self.logprob*self.one_hot,1)
        
        #the actual advantage loss is A*log(pi(a|s)) but as A=-rewards because we want 
        #to minimize then the loss is -rewards*log(pi(a|s))
        #or ¿doesn't it?
        #self.loss=tf.reduce_sum(-self.rewards*self.cross_entropies)
        self.loss=tf.reduce_mean(-self.rewards*self.cross_entropies)
        
        #training operation
        self.optimizer=tf.train.RMSPropOptimizer(learning_rate=0.000001,decay=0.99)
        self.train_op=self.optimizer.minimize(self.loss)
        
    def getBestAction(self,state):
        action=sess.run(self.best_action,feed_dict={self.observations:[state]})
        return action
    
    def getAction(self,state):
        action=sess.run(self.sample_op,feed_dict={self.observations:[state]})
        return action
    
    def train(self,npstates_batch,npactions_batch,nprewards_batch):
        feed_dict={self.observations:npstates_batch, self.actions:npactions_batch,self.rewards:nprewards_batch}
        one_hot,cross_entropies,logits,loss,_=sess.run([self.one_hot,self.cross_entropies,self.logits,self.loss,self.train_op],feed_dict=feed_dict)
        #print("logits=",logits)
        #print("cross_entropies",cross_entropies)
        #print("onehot=",one_hot)
        #print("nprewards_batch=",nprewards_batch.min())
        return loss

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
        state, reward, done, info = world.step(action)
        #collect results
        statesList.append(state)
        actionsList.append(action)
        rewardsList.append(reward)
        episode_reward+=reward
    # Process the rewards after each episode
    # Discount and Normalize reguards
    processed_rewards=discount_rewards (rewardsList,gamma)
    processed_rewards=normalize_rewards(processed_rewards)
    # Convert list to np.array
    npstates =np.vstack(statesList)
    npactions=np.vstack(actionsList)#[:,0]
    nprewards=np.vstack(processed_rewards)#[:,0]
    return npstates,npactions,nprewards,episode_reward

print("tf.version",tf.VERSION)

world = World()

pgnn=PGNetwork()

init = tf.initialize_all_variables()

world.render=False
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
        for num_episodes_batch in range(BATCH_SIZE):
            # Play a episode
            npstates,npactions,nprewards,episode_reward=playEpisode(pgnn)
            # Append data to the batch
            npstates_batch =np.vstack((npstates_batch ,npstates ))
            npactions_batch=np.vstack((npactions_batch,npactions))
            nprewards_batch=np.vstack((nprewards_batch,nprewards))
            #running mean of rewards
            episode_reward_mean=0.99*episode_reward_mean+0.01*episode_reward
            # Feedback info
            if num_episodes%1==0:
                print("{:6.0f}".format(num_episodes),
                      "eb={:6.0f}".format(num_episodes_batch),
                      "batch={:6.0f}".format(num_batches),
                      "epmn={:2.4f}".format(episode_reward_mean),
                      "eprw={:2.0f}".format(episode_reward))
            # Counts
            num_episodes+=1

        loss=pgnn.train(npstates_batch,npactions_batch[:,0],nprewards_batch[:,0])
        print("Training this batch with shape=",npstates_batch.shape,
              " loss={:2.6f}".format(loss))
        num_batches+=1
        if num_batches%10==0:
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)

