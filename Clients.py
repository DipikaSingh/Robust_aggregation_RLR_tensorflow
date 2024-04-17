import Model as FM # FM represent the Federated Model 
import Utils
import copy
import gc
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
from tensorflow.keras import backend as K
import gc
import time
from tensorflow.compat.v1.keras.backend import (
    set_session,
    clear_session,
    get_session
)
#############
from tqdm import tqdm
tf.keras.backend.clear_session()
# Enable mixed precision training
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# class Client():
#     def __init__(self, id, args, train_dataset, data_idxs=None):
#         self.id = id ### Clients id
#         self.args=args
#         # Splitting into Local datasets for the Client
#         self.local_train_dataset= Utils.DatasetSplit((train_dataset), data_idxs)
#         # poisoning of client data for backdoor attack
#         if self.id < args.num_corrupt:
#             if self.args.all_class_poisoned:
#                 self.local_train_dataset=Utils.all_class_poison_dataset(self.local_train_dataset, args ,data_idxs) # for all labels poison 
#             else:
#                 self.local_train_dataset=Utils.poison_dataset(self.local_train_dataset, args ,data_idxs) # for single label poison 
#         # size of local dataset
#         self.n_data = len(self.local_train_dataset)
        
#         #get data loader
#         self.train_loader=tf.data.Dataset.from_tensor_slices(
#             (self.local_train_dataset.images,self.local_train_dataset.targets)
#             ).batch(self.args.bs)
#         # buffer_size=self.n_data
class Client():
    def __init__(self, id, args, train_dataset, data_idxs=None):
        self.id = id ### Clients id
        self.args=args
        # Splitting into Local datasets for the Client
        self.local_train_dataset= Utils.DatasetSplit((train_dataset), data_idxs)
        # poisoning of client data for backdoor attack
        if self.id < args.num_corrupt:
            if self.args.all_class_poisoned:
                self.local_train_dataset=Utils.all_class_poison_dataset(self.local_train_dataset, args ,data_idxs) # for all labels poison 
            else:
                self.local_train_dataset=Utils.poison_dataset(self.local_train_dataset, args ,data_idxs) # for single label poison 
        # size of local dataset
        self.n_data = len(self.local_train_dataset)
        
        #get data loader
        self.train_loader=tf.data.Dataset.from_tensor_slices(
            (self.local_train_dataset.images,self.local_train_dataset.targets)
            ).shuffle(buffer_size=self.n_data,seed=111).batch(args.bs)

    @tf.function
    def run_optimizer(self, model , x , y_true,loss_obj,optimizer_obj):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss_val = loss_obj(y_true=y_true , y_pred=logits)
        grad = tape.gradient(loss_val , model.trainable_weights)
        optimizer_obj.apply_gradients(zip(grad , model.trainable_weights))
        return logits , loss_val
    
    def train_one_epoch(self, model, train_data,loss_obj,optimizer_obj,train_acc_matrix):
        losses = []
        pbar = tqdm(total=len(list(enumerate(train_data))), position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
        for batch_no , (data , label) in enumerate(train_data):
            y_pred , loss = self.run_optimizer(model , data , label,loss_obj,optimizer_obj)
            losses.append(loss)
            train_acc_matrix(label , y_pred)
            #pbar.set_description("Training loss for step %s: %.4f" % (int(batch_no), float(loss)))
            pbar.update()
        return losses
    
    def local_train(self, global_model, init_global_weights,optimizer, criterion,accuracy):
        tf.keras.backend.clear_session()
        Local_model=FM.Model(x_train_shape=(28,28,1),output_labels=self.args.n_classes).cnn()
        dims,layer_size,initial_global_model_params=Utils.parameters_to_vectors(init_global_weights)
        Local_model.compile(optimizer=optimizer, loss=criterion, metrics=accuracy)
        Local_model.set_weights(init_global_weights)
        Local_model.fit(self.train_loader, epochs=self.args.local_ep,verbose=0)
        print("Finished client={}----Local Client data number ={}".format(self.id,self.n_data))
        dims,layer_size,currr_weights=Utils.parameters_to_vectors(Local_model.get_weights())
        K.clear_session()
        del Local_model

        return currr_weights-initial_global_model_params


    def LowAPI_local_train(self, global_model, init_global_weights,optimizer, criterion,accuracy):
        # initial_global_model_params=global_model.trainable_weights
        # Initial weights are given in 
        # First flatten initial weight
        dims,layer_size,initial_global_weights_list=Utils.parameters_to_vectors(init_global_weights)
        # Set your initial  
        global_model.set_weights(init_global_weights)
        for epoch in range(self.args.local_ep):
            # self.run_optimizer(global_model , x , y,criterion,optimizer_obj)
            self.train_one_epoch(global_model, self.train_loader,criterion,optimizer,accuracy)
        print("Finished client={}----Local Client data number ={}".format(self.id,self.n_data))
    
        dims,layer_size,currr_weights_list=Utils.parameters_to_vectors(global_model.get_weights())
        return currr_weights_list-initial_global_weights_list # This will be list

    



















     # return initial_global_model_params - currr_weights
        #return currr_weights
    
    # def local_train(self, global_model, criterion,accuracy):
    #     # initial_global_model_params=global_model.trainable_weights
    #     dims,layer_size,initial_global_model_params=Utils.parameters_to_vectors(global_model.get_weights())
    #     optimizer = tf.keras.optimizers.SGD(learning_rate=self.args.client_lr,momentum=self.args.client_moment)
    #     for epoch in range(self.args.local_ep):
    #         for step, (x, y) in enumerate(self.train_loader):
    #             with tf.GradientTape() as tape:
    #                 logits = tf.clip_by_value(global_model(x),1e-10,1.0)
    #                 loss_value = criterion(y, logits)
    #             gradients = tape.gradient(loss_value, global_model.trainable_weights)
    #             accuracy.update_state(y, logits)
    #             # Update the weights of the model to minimize the loss value.
    #             # gradients = tape.gradient(loss_value, global_model.trainable_weights)
    #             optimizer.apply_gradients(zip(gradients, global_model.trainable_weights))
    #             # print(global_model.trainable_weights)
    #             # Logging the current accuracy value so far.
    #         if step % 10 == 0:
    #             print("Epoch:", epoch, "Batch:", step)
    #             print("Total running accuracy so far: %.3f" % accuracy.result())
    #     # print("Logits")
    #     # print(logits)
    #     # print("Logits")
    #     # print("LossValue:",loss_value)
        # print("Finished client={}".format(self.id))
    #     # print(global_model.trainable_weights[-1])
    #     dims,layer_size,currr_weights=Utils.parameters_to_vectors(global_model.get_weights())
    #     return currr_weights-initial_global_model_params
    #     # list(np.array(global_model.get_weights())-np.array(initial_global_model_params))

        