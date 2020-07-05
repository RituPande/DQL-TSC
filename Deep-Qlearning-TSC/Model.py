# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:31:15 2019

@author: xz01m2
"""
from  keras.models import Sequential
from  keras.layers import Dense
from  keras.optimizers import Adam

class Model:

    def __init__(self, num_states, num_actions):
        model = Sequential()
        model.add(Dense(400, input_dim= num_states))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        self.model = model
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights( self, w ):
        self.model.set_weights(w)
    
    def predict( self, state ):
        return self.model.predict(state)
    
    def fit( self, x_batch, y_batch, batch_size, verbose=0):
        self.model.fit( x_batch, y_batch, batch_size, verbose=0)
    
    def save( self, filename):
        self.model.save(filename)
  