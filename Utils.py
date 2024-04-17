from collections import defaultdict
from email.policy import default
import tensorflow as tf
import numpy as np
import random
from math import floor
import copy
import Datasets_prepartion as ds
import matplotlib.pyplot as plt
from Variables_global import args_parser

class DatasetCreate():
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset):
        self.images = dataset.images
        self.targets = dataset.targets
    

class DatasetSplit():
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.images = dataset.images[idxs]
        self.idxs = idxs
        self.targets = dataset.targets[idxs]
    
    def __len__(self):
        return len(self.idxs)


def parameters_to_vectors(X):
    ## X should be in list
    dims, layer_size ,flat_data=[],[],[]
    sum=0
    for x in X:
        dims.append(x.shape)
        sum=sum+np.prod(x.shape)
        # print(np.prod(x.shape))
        layer_size.append(sum)
        flat_data.append(x.flatten())
    return dims,layer_size, tf.convert_to_tensor(np.concatenate(flat_data, axis=0))

def vectors_to_parameters(dims,layer_size,X):
    y=np.split(X.numpy(), layer_size)[:-1]#The last element is blank, return in split function.
    Y=[]
    for i,yy in enumerate(y):
        Y.append(yy.reshape(dims[i]))
    return Y


def get_datasets(args):
    train_dataset, test_dataset = None, None
    if args.data=='fmnist':
        path='..//Datasets//Data//FashionMNIST//raw'
        train_dataset=ds.FMNIST(path,kind='train')
        test_dataset=ds.FMNIST(path,kind='t10k')
    elif args.data=='mnist':
        path='..//Datasets//Data//MNIST'
        train_dataset=ds.FMNIST(path,kind='train')
        test_dataset=ds.FMNIST(path,kind='t10k')
    
    elif args.data=='emnist':
        path='..//Datasets//Data//EMNIST//gzip'
        train_dataset=ds.EMNIST(path,kind='train',name='digits')
        test_dataset=ds.EMNIST(path,kind='test',name='digits')
    
    elif args.data=='gtsrb':
        path='..//Datasets//Data//GTSRB'
        train_dataset=ds.GTSRB(path,kind='train')
        test_dataset=ds.GTSRB(path,kind='test')

    return train_dataset, test_dataset

def distribute_data(dataset, args):
    """
    distribute data to clients 
    """

    if args.num_agents == 1:
        return {0:range(len(dataset))}

    labels_sorted_indices = np.argsort(dataset.targets)
    labels_sorted_values = dataset.targets[labels_sorted_indices]
    # print("Indices:", labels_sorted_indices[1:10])
    # print("Indices:", labels_sorted_values[1:10])
    unique, counts = np.unique(dataset.targets, return_counts=True)
    total_label = len(unique)
    num_unique_label = counts
    class_by_labels = list(zip(labels_sorted_values.tolist(), labels_sorted_indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    
    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * args.class_per_agent)
    
    slice_size = (len(dataset) // args.n_classes) // shard_size
    def chunker_list(seq, size):
            return [seq[i::size] for i in range(size)]
    
    for k, v in labels_dict.items():
            labels_dict[k] = chunker_list(v, slice_size)
    
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, args.n_classes):
            if class_ctr == args.class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j%args.n_classes][0]
                class_ctr+=1

    return dict_users       


def add_pattern_bd(args, image):
    """
    adds a trojan pattern to the image
    """
    x=copy.deepcopy(image)
    if args.pattern_type == 'plus':
    
        #x=copy.deepcopy(image)
        # x = np.array(x.squeeze())
        start_idx = 2
        size = 5
        # vertical line  
        for i in range(start_idx, start_idx+size):
            # print(i)
            x[i, start_idx] = 1

        # horizontal line
        for i in range(start_idx-size//2, start_idx+size//2 + 1):
            x[start_idx+size//2, i] = 1

    elif args.pattern_type == 'square':

        for i in range(23,27):
            for j in range(23,27):
                x[i,j] = 1
    
    elif args.pattern_type == 'yellow_stick':
        # Trigger is located on the stop sign (which is located at the center of images)
        pos_x = int(image.shape[1]/2)
        pos_y = int(image.shape[1]/2)
        trigger = np.stack([np.ones(shape=(3,3)), np.ones(shape=(3,3))
        ,np.zeros(shape=(3,3))],axis=2)
        # create temp mask of same size as input image with all 1's
        mask1 = np.ones(image.shape) 
        mask1[pos_y : pos_y + trigger.shape[0], pos_x : pos_x + trigger.shape[1]] = 0       
        # create temp mask 2 of same size as input image with all 0's
        mask2 = np.zeros(image.shape)
        mask2[pos_y : pos_y + trigger.shape[0], pos_x : pos_x + trigger.shape[1]] = trigger
        x=image * mask1 + mask2

    elif args.pattern_type == 'apple':
        trojan = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(28,28), interpolation=cv2.INTER_CUBIC)

        # print(trojan[:,:,np.newaxis].shape)
        x = x + trojan[:,:,np.newaxis]/255.0

    return x



def poison_dataset(DataO,args, data_idxs=None, poison_all=False,if_poison_idxs_rtn=False):
    local_Data=copy.deepcopy(DataO)

    ######## store the remove_idxs: if you want to remove original target class
    remove_idxs=np.nonzero(local_Data.targets==args.target_class)[0].flatten().tolist()

    #### I have taken poison_idxs for knowing about the poisoned images
    all_idxs=np.nonzero(local_Data.targets==args.base_class)[0].flatten().tolist()
    ### if you want to remove original target class completely

    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
    
  
    poison_frac = 1 if poison_all else args.poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs))) 

    if not poison_idxs:
        return local_Data
        
    for idx in poison_idxs:
        clean_img=local_Data.images[idx,]
        bd_img =add_pattern_bd(args, clean_img)
        local_Data.images[idx,]=bd_img
        local_Data.targets[idx]=args.target_class
        
    
    if if_poison_idxs_rtn:
        return local_Data,remove_target_class(local_Data,remove_idxs), [poison_idxs]
    else:
        return local_Data


def all_class_poison_dataset(DataO,args, data_idxs=None, poison_all=False,if_poison_idxs_rtn=False):
    local_Data=copy.deepcopy(DataO)

    ######## store the remove_idxs: if you want to remove original target class
    remove_idxs=np.nonzero(local_Data.targets==args.target_class)[0].flatten().tolist()

    all_poison_idxs=[]
    for base_class in args.array_base_class:
        # if base_class==args.target_class:
        #     continue
        all_idxs=np.nonzero(local_Data.targets==base_class)[0].flatten().tolist()
        ### if you want to remove original target class completly
        if data_idxs != None:
            all_idxs = list(set(all_idxs).intersection(data_idxs))
        
        poison_frac = 1 if poison_all else args.poison_frac
        poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs))) 
        all_poison_idxs.append(poison_idxs)
        if not poison_idxs:
            continue
        for idx in poison_idxs:
            clean_img=local_Data.images[idx,]
            bd_img =add_pattern_bd(args, clean_img)
            local_Data.images[idx,]=bd_img
            local_Data.targets[idx]=args.target_class

    if if_poison_idxs_rtn:
        return local_Data,remove_target_class(local_Data,remove_idxs), all_poison_idxs
    else:
        return local_Data



def remove_target_class(DataI,remove_idxs=None):
    local_Data=copy.deepcopy(DataI)
    local_Data.targets=np.delete(local_Data.targets,remove_idxs, axis=0)
    local_Data.images=np.delete(local_Data.images,remove_idxs, axis=0)
    print('Remove=',len(local_Data.targets))
    return local_Data

# def get_loss_n_accuracy(model, criterion, data_loader, num_classes=10):



def plot_figure(x):
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.imshow(x)
    plt.show()
    

 
if __name__ == "__main__":
    train, test = get_datasets()
    distribute_data(train)

# train, test = get_datasets()
# poison_dataset(train)
# index = 55519
# plot_figure(train[index,])
# plt.show()
