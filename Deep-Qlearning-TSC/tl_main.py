# -*- coding: utf-8 -*-

import os
import sys
from  Model import Model
import glob
import traci
from SumoEnv import SumoEnv
import numpy as np
import random
from TrafficGenerator import TrafficGenerator
from collections import deque
import utils 
from keras.models import load_model



#import traci.constants as tc
class TLAgent:
    
    def __init__( self, env, traffic_gen, max_steps, total_episodes):
            
        self.env = env
        self.traffic_gen = traffic_gen
        self.total_episodes = total_episodes
        self.discount = 0.75
        self.epsilon = 0.9
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 100
        self.num_states = 80
        self.num_actions = 4
        self.num_experiments = 5
        # phases are in same order as specified in the .net.xml file
        self.PHASE_NS_GREEN = 0  # action 0 code 00
        self.PHASE_NS_YELLOW = 1
        self.PHASE_NSL_GREEN = 2  # action 1 code 01
        self.PHASE_NSL_YELLOW = 3
        self.PHASE_EW_GREEN = 4  # action 2 code 10
        self.PHASE_EW_YELLOW = 5
        self.PHASE_EWL_GREEN = 6  # action 3 code 11
        self.PHASE_EWL_YELLOW = 7
        
        self.green_duration = 10
        self.yellow_duration = 4
        
        self.init_epoch = 0
        self.QModel = None
        self.TargetQModel = None
        self._load_models()
        self.max_steps = max_steps
        
        self.reward_store = np.zeros((self.total_episodes, ))
        self.intersection_queue_store = np.zeros((self.total_episodes,) )
        
       
      
    def _load_models( self ) :
        
        self.QModel = Model(self.num_states, self.num_actions )
        self.TargetQModel = Model(self.num_states, self.num_actions)
        qmodel_file_name = glob.glob('./qmodel*')
       
        if  qmodel_file_name:
            qmodel_fd = open(qmodel_file_name[0], 'r')
                   
            if (qmodel_fd is not None):
                self.init_epoch = utils._get_init_epoch(qmodel_fd.name)
                self.QModel = load_model(qmodel_fd.name)
                self.TargetQModel = load_model(qmodel_fd.name)
            
        return   self.QModel, self.TargetQModel                     
            
    def _preprocess_input( self, state ):
        state = np.reshape(state, [1, self.num_states])
        return state
    
    def _add_to_replay_buffer( self, curr_state, action, reward, next_state, done ):
        self.replay_buffer.append((curr_state, action, reward, next_state, done))
        
    def _sync_target_model( self ):
        self.TargetQModel.set_weights( self.QModel.get_weights()) 
        
    def _replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample( self.replay_buffer, min(len(self.replay_buffer), self.batch_size)) 
        
        for i in range( len(mini_batch)):
            curr_state, action, reward, next_state, done = mini_batch[i]
            y_target = self.QModel.predict(curr_state) # get existing Qvalues for the current state
            y_target[0][action] = reward if done else reward + self.discount*np.max(self.TargetQModel.predict(next_state)) # modify the qvalues for the action perfomrmed to get the new target 
            x_batch.append(curr_state[0])
            y_batch.append(y_target[0])
        
        self.QModel.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        
        
    def _agent_policy( self, episode, state ):
        epsilon = 1 - episode/self.total_episodes
        choice  = np.random.random()
        if choice <= epsilon:
            action = np.random.choice(range(self.num_actions))
        else:
            action =  np.argmax(self.QModel.predict(state))
                
        return action
        
     # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase("TL", self.PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("TL", self.PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("TL", self.PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("TL", self.PHASE_EWL_GREEN)

            
    def train( self ):
        
        curr_state = self.env.start()
   
        for e in range(self.total_episodes):
            self.traffic_gen.generate_routefile(7)
            curr_state = self._preprocess_input( curr_state)
            old_action =  None
            done = False # whether  the episode has ended or not
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            while not done:
                    
                action = self._agent_policy( e,curr_state)
                yellow_reward = 0
                    
                if old_action!= None and old_action != action:
                    self._set_yellow_phase(old_action)
                    yellow_reward, _ , _ = self.env.step(self.yellow_duration)
                   
                self._set_green_phase(action)
                reward, next_state, done = self.env.step(self.green_duration)
                reward += yellow_reward
                    #print("reward-main={}".format(reward))
                next_state = self._preprocess_input( next_state )
                self._add_to_replay_buffer( curr_state, action, reward, next_state, done )
                self._replay()
                curr_state = next_state
                old_action = action
                sum_intersection_queue += self.env.get_intersection_q_per_step()
                if reward < 0:
                    sum_neg_rewards += reward
                    
          
            self._save_stats(e, sum_intersection_queue,sum_neg_rewards)
            self.QModel.save('qmodel_{}.hd5'.format(e))
            if e != 0:
                os.remove('qmodel_{}.hd5'.format(e-1))
            curr_state = self.env.reset()   # reset the environment before every episode
            print('Epoch {} complete'.format(e))
        
    def execute( self):
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
    def _save_stats(self, episode, sum_intersection_queue_per_episode, sum_rewards_per_episode):
        self.reward_store[episode] = sum_rewards_per_episode
        self.intersection_queue_store[episode] = sum_intersection_queue_per_episode  
  
        
if __name__ == "__main__":

    # --- TRAINING OPTIONS ---
    training_enabled = True
    gui = False
 
<<<<<<< HEAD
  
=======
   
>>>>>>> parent of 1414c80... Changes required to upload
    # ----------------------

    # attributes of the agent
  
    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = 'sumo.exe'
    else:
        sumoBinary ='sumo-gui.exe'

    # initializations
    max_steps = 5400  # seconds = 1 h 30 min each episode
    total_episodes = 100
    num_experiments = 5

    traffic_gen = TrafficGenerator(max_steps)
   
        
    if training_enabled:
       

        reward_store = np.zeros((num_experiments,total_episodes))
        intersection_queue_store = np.zeros((num_experiments,total_episodes))
        for experiment in range(num_experiments):
            env = SumoEnv(sumoBinary,max_steps )
            tl = TLAgent( env, traffic_gen, max_steps, total_episodes )
            tl.train()
            reward_store[experiment,:] = tl.reward_store
            intersection_queue_store[experiment,:] =  tl.intersection_queue_store
            del env
            del tl
            print('Experiment {} complete.........'.format(experiment))
        utils.plot_rewards(reward_store)
        utils.plot_intersection_queue_size( intersection_queue_store)
        print(reward_store)
        print(intersection_queue_store)
        
        
    
    
