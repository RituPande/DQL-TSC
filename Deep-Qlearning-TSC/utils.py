# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:55:20 2020

@author: XZ01M2
"""

import matplotlib.pyplot as plt
import numpy as np 
import glob

def get_file_names():
     qmodel_file_name = glob.glob('./qmodel*')
     stats_file_name = glob.glob('./stats*')
     
     if not qmodel_file_name:
         qmodel_file_name = ''
     else:
         qmodel_file_name = qmodel_file_name[0]
    
     if not stats_file_name:
         stats_file_name = ''
     else:
         stats_file_name = stats_file_name[0]

     return qmodel_file_name, stats_file_name

def get_init_epoch( filename,total_episodes, learn = True ):
    
    if filename and learn:
        index = filename.find('_')
        exp = int(filename[index+1]) 
        epoch= int(filename[index+3])
        if epoch < total_episodes -1:
            epoch +=1
        else:
            epoch = 0
            exp +=1
        
    else:
        exp=0
        epoch = 0
    return exp , epoch

def get_stats(stats_filename, num_experiments, total_episodes, learn = True):
    
    if stats_filename and learn:
        stats =np.load(stats_filename, allow_pickle = True)[()]
    else:
        reward_store = np.zeros((num_experiments,total_episodes))
        intersection_queue_store = np.zeros((num_experiments,total_episodes))
        stats = {'rewards': reward_store, 'intersection_queue': intersection_queue_store }

    return stats
    
def plot_rewards(reward_store):
    
    x = np.mean(reward_store, axis = 0 )
    plt.plot( x , label = "Cummulative negative wait times") 
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative negative wait times') 
    plt.title('Cummulative negative wait times across episodes') 
    plt.legend() 
    plt.show() 
    
def plot_intersection_queue_size( intersection_queue_store):
    
    x = np.mean(intersection_queue_store, axis = 0 )
    plt.plot(x, label = "Cummulative intersection queue size ", color='m') 
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative intersection queue size') 
    plt.title('Cummulative intersection queue size across episodes') 
    plt.legend() 
    plt.show() 