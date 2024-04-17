from Variables_global import args_parser
import Utils
import tensorflow as tf
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [ft.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
import matplotlib.pyplot as plt
#########
import Model as FM # FM represent the Federated Model  
from Clients import Client
from Aggregation import *
#############
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math
import copy
import os
import json
import gc
import time
import os
import psutil
from tensorflow.compat.v1.keras.backend import (
    set_session,
    clear_session,
    get_session
)
tf.keras.backend.clear_session()
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info().rss/1e9  # memory use in GB...I think
print('memory use at Starting:', memoryUse)
# Enable mixed precision training
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
class Main:
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        # Make folder to save the results:
        #Directory
        self.dir_path = os.path.join('/home/users/dsingh/FL_Backdoor_V6/SIGN_MNIST_plus',args.case)
        if os.path.exists(self.dir_path):
            print('No path is created this time: Check the case name in Global_variables')
        else:
            os.makedirs(self.dir_path)

        ##### Dictionary for storing Loss, Accuracy,
        self.dict_output = {'Validation_Loss':[],'Validation_Acc':[],'Round':[],
        'Poisoned_Loss':[],'Poisoned_Acc':[],
        'Poisoned_class_Acc':[],'Validation_class_Acc':[],'Poisoned_Accuracy_Loss':[],'Poisoned_Accuracy_Acc':[],
        'Poisoned_Accuracy_class_Acc':[]}
        #####
        self.train_dataset, self.val_dataset = Utils.get_datasets(args)
        ### Distributed in user groups for train datasets
        self.user_groups = Utils.distribute_data(self.train_dataset,self.args)
        # self.user_groups contains dict of client groups number (ex: 0,1,2..num_clients) 
        ##- And values are sorted indexes of each labels.
        #-------- Test data sets-----

        ##self.poisoned_dataset_without_target_class_rm this variable is nothing
        ### just to make sure that class has been poisoned and 
        #### the purpose of this variable to keep the index same
        ##### so I can compare Images with self.val_dataset
        if self.args.all_class_poisoned:
            self.poisoned_dataset_without_target_class_rm,self.poisoned_dataset,self.pois_idxs=Utils.all_class_poison_dataset(self.val_dataset, 
        self.args,poison_all=True,if_poison_idxs_rtn=True)
        
        else:
            self.poisoned_dataset_without_target_class_rm,self.poisoned_dataset,self.pois_idxs=Utils.poison_dataset(self.val_dataset, 
        self.args,poison_all=True,if_poison_idxs_rtn=True)
        # Make sure poisoned_val_dataset class set length
        print("poisoned_val_dataset:",len(self.poisoned_dataset.targets))
        ### all labels are poisoned expect target

        if self.args.save_figure:
            self.plot_images()
        self.val_batched = tf.data.Dataset.from_tensor_slices((self.val_dataset.images, 
        self.val_dataset.targets)).shuffle(buffer_size=len(self.val_dataset.targets),seed=111).batch(len(self.val_dataset.targets))

        self.poisoned_dataset_batched_with_actual_target = tf.data.Dataset.from_tensor_slices((self.poisoned_dataset_without_target_class_rm.images, 
        self.val_dataset.targets)).shuffle(buffer_size=len(self.val_dataset.targets),seed=111).batch(len(self.val_dataset.targets))

        self.poisoned_dataset_batched = tf.data.Dataset.from_tensor_slices((self.poisoned_dataset_without_target_class_rm.images, 
        self.poisoned_dataset_without_target_class_rm.targets)).shuffle(buffer_size=len(self.poisoned_dataset_without_target_class_rm.targets),seed=111).batch(len(self.poisoned_dataset_without_target_class_rm.targets))
    
    def Validation_check(self, model, cce,comm_round):
        ## Making validation test batched 
        
        print("********************************************")
        print("***********Validation sets******************")
        A,L,P_A=self.Accuracy_check(self.val_batched,model,cce, comm_round,labels=np.unique(self.val_dataset.targets))
        # results = model.evaluate(val_batched)
        print("********************************************")
        print("============================================")
        self.dict_output['Validation_Loss'].append(L)
        self.dict_output['Validation_Acc'].append(A)
        self.dict_output['Validation_class_Acc'].append(P_A)


    def poisoned_check_Accuracy(self, model, cce,comm_round):
        ## Making poisoned datasets test batched 
       
        print("********************************************")
        if self.args.all_class_poisoned:
            print("***********poisoned_Accuracy: All classes******************")
        else:
            print("***********poisoned_Accuracy: one class******************")
        A,L,P_A=self.Accuracy_check(self.poisoned_dataset_batched_with_actual_target,model,cce, comm_round,labels=np.unique(self.val_dataset.targets))
        # results = model.evaluate(poisoned_dataset_batched)
        print("********************************************")
        print("============================================")
        print("Evaluate on test data")
        # print("test loss, test acc:", results)
        self.dict_output['Poisoned_Accuracy_Loss'].append(L)
        self.dict_output['Poisoned_Accuracy_Acc'].append(A)
        self.dict_output['Poisoned_Accuracy_class_Acc'].append(P_A)
        
    
    def poisoned_check(self, model, cce,comm_round):
        ## Making poisoned datasets test batched 
        print("********************************************")
        if self.args.all_class_poisoned:
            print("***********poisoned: All classes******************")
        else:
            print("***********poisoned: one class******************")
        A,L,P_A=self.Accuracy_check(self.poisoned_dataset_batched,model,cce, comm_round,labels=np.unique(self.val_dataset.targets))
        # results = model.evaluate(poisoned_dataset_batched)
        print("********************************************")
        print("============================================")
        print("Evaluate on test data")
        # print("test loss, test acc:", results)
        self.dict_output['Poisoned_Loss'].append(L)
        self.dict_output['Poisoned_Acc'].append(A)
        self.dict_output['Poisoned_class_Acc'].append(P_A)
    
    def memory(self, name='Deep'):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info().rss/1e9  # memory use in GB...I think
        print(name)
        print('memory use:', memoryUse)
        
    
    def run(self):
        tf.keras.backend.clear_session()
        global_model = FM.Model(x_train_shape=self.train_dataset.images[0,].shape,output_labels=self.args.n_classes).cnn()
        print(global_model.summary())
        agents, agent_data_sizes = [], {}
        for _id in range(0, self.args.num_agents):
            agent = Client(_id, self.args , self.train_dataset, self.user_groups[_id])
            agents.append(agent)
            agent_data_sizes[_id] = agent.n_data
        
        ### CNN model input
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.args.client_lr,momentum=self.args.client_moment)
        accuracy=tf.keras.metrics.SparseCategoricalAccuracy()


        #aggregation server
        
        dims,layer_size,global_weights_list = Utils.parameters_to_vectors(global_model.get_weights())
        aggregator = Aggregation(agent_data_sizes,self.args,n_params=len(global_weights_list))

        for rnd in tqdm(range(1, self.args.rounds+1)):
            print("Round={}".format(rnd))
            # This parameters_to_vectors I defined to get the vector (flatten List) 
            ## weights for aggregations.
            dims,layer_size,global_weights_list = Utils.parameters_to_vectors(global_model.get_weights())
            global_weights_params = global_model.get_weights()
            tf.print("Size of initial weights")
            tf.print(tf.size(global_weights_list))
            agent_updates_dict = {}
            for agent_id in np.random.choice(self.args.num_agents, math.floor(self.args.num_agents*self.args.agent_frac), replace=False):
                update = agents[agent_id].LowAPI_local_train(global_model,global_weights_params,optimizer,criterion,accuracy)
                agent_updates_dict[agent_id] = update
                self.memory(name='Each client after')
            average_weights=aggregator.aggregate_updates(global_weights_list,agent_updates_dict, rnd)
            global_model.set_weights(Utils.vectors_to_parameters(dims,layer_size,(average_weights)))
            ### Validation data sets are here
            self.Validation_check(global_model, criterion, rnd )
            self.poisoned_check(global_model,criterion, rnd )
            self.poisoned_check_Accuracy(global_model,criterion, rnd )
            self.dict_output['Round'].append(rnd)
            #global_model.save_weights(os.path.join(self.dir_path,'Comm_Round_'+str(int(rnd))+'_model_weights'+'.h5')) #save the model weights for each round
            self.memory(name='Each Round after')
            self.reset_memory()
            tf.keras.backend.clear_session()
        
        self.plot_results()
        #### dictionary save
        with open(os.path.join(self.dir_path,'Result_Data.json'), "w") as outfile:
            json.dump(self.dict_output, outfile)

        # self.model_explainable(global_model,self.train_dataset.images, self.val_dataset.images, name='Validation')
        # self.model_explainable(global_model,self.train_dataset.images, self.poisoned_dataset.images, name='Poisoned')
        exe_time = time.time() - self.start_time
        final_time = exe_time / 60
        with open(os.path.join(self.dir_path,'commandline_args.txt'), 'a') as f:
            f.write(f'----Execution time------{final_time} minutes')
        # self.memory()
        # self.reset_memory(model=global_model)
        
    
    def Accuracy_check(self, Bacth_data_XY,  model, cce, comm_round,labels=[0,1,2,3,4,5,6,7,8,9]):
        for(X, Y) in Bacth_data_XY:
            # cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            #tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            logits = model.predict(X)
            # logits = tf.clip_by_value(model.predict(X),1e-10,1.0)
            loss = cce(Y, logits)
            sff=tf.argmax(logits, axis=1)
            acc = accuracy_score(Y,tf.argmax(logits, axis=1))
            print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
            matrix=confusion_matrix(Y, tf.argmax(logits, axis=1), labels=labels)
            print(matrix)
            per_class_accuracy=matrix.diagonal()/matrix.sum(axis=1)
            print(per_class_accuracy)
        return float(acc),float(loss),per_class_accuracy.tolist()
    
    def plot_images(self):
        """_summary_

        Args:
            x (_type_): 
            y (_type_): 1-D array
            label (int, optional): This uses for plotting a specific label. Defaults to 2.
        """
        N_labels=len(self.pois_idxs)
        if N_labels>5:
            N_labels=6
        fig, axs = plt.subplots(N_labels,self.args.n_figure, figsize=(10, 6), constrained_layout=True, dpi=600)
        for nlabel in np.arange(N_labels)[0::2]:
            for i,ax in enumerate(axs[nlabel,:]):
                ax.imshow(self.val_dataset.images[self.pois_idxs[nlabel][i]],aspect='equal', cmap='gray')
            for i,ax in enumerate(axs[nlabel+1,:]):
                ax.imshow(self.poisoned_dataset_without_target_class_rm.images[self.pois_idxs[nlabel][i]],aspect='equal',cmap='gray')
        fig.savefig(os.path.join(self.dir_path,'Images_with_and_without_poisoned.png'))
        plt.close()
        
    def plot_results(self):
        fig, axs = plt.subplots(1,2, figsize=(10, 5), constrained_layout=True)
        # axs[0].plot(self.dict_output['Round'],self.dict_output['Poisoned_Loss'],label='Poisoned_Loss',linewidth=1,
        # c='r')
        axs[0].plot(self.dict_output['Round'],self.dict_output['Validation_Loss'],label='Validation_Loss',linewidth=1,
        c='b')
        axs[0].plot(self.dict_output['Round'],self.dict_output['Poisoned_Loss'],label='Backdoor_Loss',linewidth=1,
        c='g')
        ###
        axs[0].set_xlabel('Communication_Rounds')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        ### Accuracy 
        axs[1].plot(self.dict_output['Round'],self.dict_output['Validation_Acc'],label='Validation_Acc',linewidth=1,
        c='b')
        # axs[1].plot(self.dict_output['Round'],self.dict_output['Poisoned_Acc'],label='Poisoned_Acc',linewidth=1,
        # c='b',)
        axs[1].plot(self.dict_output['Round'],np.array(self.dict_output['Poisoned_class_Acc'], dtype=float)[:,self.args.target_class],label='Backdoor_Acc',linewidth=1,
        c='g')
        axs[1].legend()
        axs[1].set_xlabel('Communication_Rounds')
        axs[1].set_ylabel('Accuracy')
        # axs[1].set_ylabel('Accuracy')

        # ### All class accuracy for validation
        # axs[2].plot(self.dict_output['Round'],self.dict_output['Validation_Acc'],label='Validation_Acc',linewidth=1,
        # c='r')
        # axs[1].plot(self.dict_output['Round'],self.dict_output['Poisoned_Acc'],label='Poisoned_Acc',linewidth=1,
        # c='b',)
        # axs[1].plot(self.dict_output['Round'],self.dict_output['Poisoned_class_Acc'][:,self.args.target_class],label='Poisoned_Acc for target',linewidth=1,
        # c='b')
        # axs[1].set_xlabel('Communication_Rounds')
        # axs[1].set_ylabel('Accuracy')


        ### Accuracy
        if self.args.all_class_poisoned:
            name_plot='Result_all_class_poisoned.png'
        else:
            name_plot='Result_signal_class_poisoned.png'
        fig.savefig(os.path.join(self.dir_path,name_plot))
    
    def reset_memory(self,model=None):

        '''
        reset tensorflow graph, clear gpu memory, garbage collect.
        '''
        tf.keras.backend.clear_session()
        # sess = get_session()
        # clear_session()
        # sess.close()

        try:
            del model

        except:
            pass

        # #
        # # reset config: similar to when session created.
        # #
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # config.gpu_options.visible_device_list = '0'
        # set_session(tf.tensorflow.Session(config=config))
        # config = tf.compat.v1.ConfigProto()
        # # config.gpu_options.per_process_gpu_memory_fraction =0.222
        # session = tf.compat.v1.Session(config=config)
        # set_session(session)
        gc.collect()

       
    def print_exp_details(self):
        with open(os.path.join(self.dir_path,'commandline_args.txt'), 'w') as f:
            f.write('======================================\n')
            f.write(f'    Case Number: {self.args.case}\n')
            f.write(f'    Data: {self.args.data}\n')
            f.write(f'    Global Rounds: {self.args.rounds}\n')
            f.write(f'    Aggregation Function: {self.args.aggr}\n')
            f.write(f'    Pattern_type: {self.args.pattern_type}\n')
            f.write(f'    Number of agents: {self.args.num_agents}\n')
            f.write(f'    Fraction of agents: {self.args.agent_frac}\n')
            f.write(f'    Batch size: {self.args.bs}\n')
            f.write(f'    Epochs: {self.args.local_ep}\n')
            f.write(f'    Client_LR: {self.args.client_lr}\n')
            f.write(f'    Server_LR: {self.args.server_lr}\n')
            f.write(f'    Client_Momentum: {self.args.client_moment}\n')
            f.write(f'    RobustLR_threshold: {self.args.robustLR_threshold}\n')
            # print(f'    Noise Ratio: {self.args.noise}')
            f.write(f'    Number of corrupt agents: {self.args.num_corrupt}\n')
            f.write(f'    Poison Frac: {self.args.poison_frac}\n')
            if self.args.all_class_poisoned:
                f.write(f'    Base class array: {self.args.array_base_class}\n')
            else:
                f.write(f'    Base class: {self.args.base_class}\n')
            f.write(f'    Target class: {self.args.target_class}\n')
            f.write(f'    class_per_agent: {self.args.class_per_agent}\n')
            f.write(f'    n_classes: {self.args.n_classes}\n')
            f.write('======================================\n')
            # f.write(f'    Data: {self.args.data}\n')
            # f.write(f'    Data: {self.args.data}\n')
        print('======================================')
        print(f'    Data: {self.args.data}')
        print(f'    Global Rounds: {self.args.rounds}')
        print(f'    Aggregation Function: {self.args.aggr}')
        print(f'    Number of agents: {self.args.num_agents}')
        print(f'    Fraction of agents: {self.args.agent_frac}')
        print(f'    Batch size: {self.args.bs}')
        print(f'    Epochs: {self.args.local_ep}')
        print(f'    Client_LR: {self.args.client_lr}')
        print(f'    Server_LR: {self.args.server_lr}')
        print(f'    Client_Momentum: {self.args.client_moment}')
        print(f'    RobustLR_threshold: {self.args.robustLR_threshold}')
        # print(f'    Noise Ratio: {self.args.noise}')
        print(f'    Number of corrupt agents: {self.args.num_corrupt}')
        print(f'    Poison Frac: {self.args.poison_frac}')
        if self.args.all_class_poisoned:
            print(f'    Base class array: {self.args.array_base_class}')
        else:
            print(f'    Base class: {self.args.base_class}')
        print(f'    Target class: {self.args.target_class}')
        print(f'    class_per_agent: {self.args.class_per_agent}\n')
        print(f'For non iid, provide class_per_agent =2 or 3\n')
        print(f'    n_classes: {self.args.n_classes}\n')

if __name__=='__main__':
    #start_time = time.time()
    args = args_parser()
    Try_sess=Main(args)
    
    Try_sess.print_exp_details()
    Try_sess.run()

   
        