import numpy as np
from copy import deepcopy
from Model import*
import tensorflow as tf
import Utils
#Experiment Hyperparamters:
#Dataset: EMNIST, MNIST, FMNIST, GTSRB
#Model: 5 layer CNN model as mentioned in the paper. 
#However, we decrease the filter size of both convolution layer to 2x2 to increase the backdoor success with small pattern trojan. 
#Number of communication rounds:200
#Number of participants:10
#Fraction of corrupt agents: 30%
#Fraction of trojan samples in corrupt agent dataset: 0.5
#Fraction of selected agent for training in a round: 1
#epochs: 5
#batch_size: 16
#Local learning rate: 0.01
#Server learning rate: 1.0
#RLR threshold: 5


class Aggregation():
    
    def __init__(self, agent_data_sizes,args,n_params=None):
        """
        
        self.agent_data_sizes (__dictionary__): key: clientID and value: indexes of data of the client
        self.server_lr (__float__): server learning rate. Default is 1.0
        self.n_params : length of weights from all layers   
        self.robustLR_threshold (__int__): Threshold for defense. If total number of participants are 100 and number of corrupt participants 
        is between 10 to 40, use robustLR_threhold=50. If number of corrupt participants are more than 50%, increase the threshold to 70.
        Similarly, if number of participants are 10 and corrupt participants is between 1 to 4, use threshold=5 and for more than 5 corrupt partipants,
        increase the threshold to 7.
        Default is 0 in case of no defense.
        self.aggr (__str__): aggregation function. Default is avg

        """
        self.agent_data_sizes = agent_data_sizes
        self.server_lr = args.server_lr
        self.n_params = n_params       
        self.robustLR_threshold = args.robustLR_threshold
        self.aggr = args.aggr
        
    
    def aggregate_updates(self, cur_global_params,agent_updates_dict, cur_round):
        """function computes the aggregation of updates from all the clients

        Args:
            cur_global_params: current model weights
            agent_updates_dict (_dictionary_): dictionary of updates where key is: agent id and value is indexes of data
            cur_round (__int__): current communication round
            

        Returns:
           the function returns new parameters of the model
        """
        
        
        lr_vector = tf.convert_to_tensor([self.server_lr]*self.n_params)  #to calculate number of parameters and then multiply in LR
        
        # adjust lr_vector if robust_LR is more than 0
        if self.robustLR_threshold > 0:
            print("The robustLR_threshold:",self.robustLR_threshold)
            lr_vector = self.compute_robustLR(agent_updates_dict)

        #aggregated_updates = 0
        if self.aggr=='avg':
            aggregated_updates = self.agg_avg(agent_updates_dict)
        
        # update the model parameter
        new_global_params =  cur_global_params + lr_vector*aggregated_updates
        
        return new_global_params 

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg algorithm
        
        """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        
        print("Total weights:",sm_updates / total_data)
        
        return  sm_updates / total_data
    
    def compute_robustLR(self, agent_updates_dict):
        """function which computes robustLR 
       
        """
        agent_updates_sign = [tf.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = tf.abs(sum(agent_updates_sign)).numpy()
        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr                                            
        return tf.convert_to_tensor(sm_of_signs)
    
    