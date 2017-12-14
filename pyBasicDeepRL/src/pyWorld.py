'''
Created on Nov 30, 2017

@author: Francisco Dominguez
'''
import numpy as np
import gym

class World:
    def __init__(self):
        self.env=gym.make('Pong-v0')
        self.reset()
        self.render=False
    def reset(self):
        self.currObservation=self.env.reset()
        self.prepCurrObservation=self.preprocess(self.currObservation)
        self.state=self.prepCurrObservation
        self.prevObservation=None
        self.prepPrevObservation=None
        return self.state
    # Some times gym actions (internal) are duplicated and don't coincide with model ones (external)
    # for instance Pong-v0 has 6 acion but actually there are only 3
    # action ==0 : no action
    # action ==2 or action==4 up
    # aaction==3 or action==5 down
    def getInternalAction(self,action):
        if action==0:
            return 0
        if action==1:
            return 2
        if action==2:
            return 3
    def getExternalAction(self,action):
        if action==0 or action==1:
            return 0
        if action==2 or action==4:
            return 1
        if action==3 or action==5:
            return 2    # return next state given an action
    # it could be thet same than a observation or not
    def step(self,actionExternal):
        actionI=self.getInternalAction(actionExternal)
        self.prevObservation=self.currObservation
        self.prepPrevObservation=self.prepCurrObservation
        if self.render:
            self.env.render()
        self.currObservation, reward, done, info=self.env.step(actionI) # take a random action
        self.prepCurrObservation=self.preprocess(self.currObservation)
        self.state=self.prepCurrObservation-self.prepPrevObservation
        return self.state, reward, done, info
    def stepSample(self):
        actionI=self.env.action_space.sample()
        self.prevObservation=self.currObservation
        self.prepPrevObservation=self.prepCurrObservation
        if self.render:
            self.env.render()
        self.currObservation, reward, done, info=self.env.step(actionI) # take a random action
        self.prepCurrObservation=self.preprocess(self.currObservation)
        self.state=self.prepCurrObservation-self.prepPrevObservation
        return self.state, reward, done, info
    def actionSample(self):
        actionI = self.env.action_space.sample()
        action=self.getExternalAction(actionI)
        return action
    
    #Preprocessing methods
    def downsample(self,image):
        # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
        return image[::2, ::2, :]
    
    def remove_color(self,image):
        """Convert all color (RGB is the third dimension in the image)"""
        return image[:, :, 0]
    
    def remove_background(self,image):
        image[image == 144] = 0
        image[image == 109] = 0
        return image
    
    def preprocess(self,input_observation):
        """ convert the 210x160x3 uint8 frame into a 6400 float vector """
        processed_observation = input_observation[35:195] # crop
        processed_observation = self.downsample(processed_observation)
        processed_observation = self.remove_color(processed_observation)
        processed_observation = self.remove_background(processed_observation)
        processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
            # Convert from 80 x 80 matrix to 1600 x 1 matrix
        processed_observation = processed_observation.astype(np.float).ravel()
        return processed_observation
   
   