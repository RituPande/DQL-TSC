# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:55:20 2020

@author: XZ01M2
"""

import matplotlib.pyplot as plt 

def _get_init_epoch( filename ):
    
    index = filename.find('_')
    str_epoch = filename[index+1]
    return int(str_epoch)

    
def plot_rewards(reward_store):
    
    plt.plot(reward_store, label = "Cummulative negative wait times") 
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative negative wait times') 
    plt.title('Cummulative negative wait times across episodes') 
    plt.legend() 
    plt.show() 
    
def plot_intersection_queue_size( avg_intersection_queue_store):
   
    plt.plot(avg_intersection_queue_store, label = "Cummulative intersection queue size ", color='m') 
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative intersection queue size') 
    plt.title('Cummulative intersection queue size across episodes') 
    plt.legend() 
    plt.show() 