import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--case', type=str, default='case_1',
                        help="dataset we want to train on")

    parser.add_argument('--data', type=str, default='mnist',
                        help="dataset we want to train on")

    parser.add_argument('--num_agents', type=int, default=100,
                        help="number of agents:K")
    
    parser.add_argument('--agent_frac', type=float, default=1,
                        help="fraction of agents per round:C")
    
    parser.add_argument('--num_corrupt', type=int, default=30,
                        help="number of corrupt agents")
    
    parser.add_argument('--rounds', type=int, default=200,
                        help="number of communication rounds:R")
    
    parser.add_argument('--aggr', type=str, default='sign', 
                        help="aggregation function to aggregate agents' local weights")

    parser.add_argument('--pattern_type', type=str, default='plus', 
                        help="shape of bd pattern")
    
    parser.add_argument('--local_ep', type=int, default=5,
                        help="number of local epochs:E")
    
    parser.add_argument('--bs', type=int, default=16,
                        help="local batch size: B")
    
    parser.add_argument('--client_lr', type=float, default=0.01,
                        help='clients learning rate')
    
    parser.add_argument('--client_moment', type=float, default=0.9,
                        help='clients momentum')
    
    parser.add_argument('--server_lr', type=float, default=1.0,
                        help='servers learning rate for signSGD')
    
    parser.add_argument('--base_class', type=int, default=5, 
                        help="base class required in case of single class poison")

    parser.add_argument('--all_class_poisoned', type=str, default=True, 
                        help="Make all class poisoned expect target class") 

    parser.add_argument('--array_base_class', type=int, default=list(np.arange(0,10,1)), 
                        help="define list of base class for backdoor attack with target class")
                        
    parser.add_argument('--target_class', type=int, default=8, 
                        help="target class for backdoor attack")
    
    parser.add_argument('--poison_frac', type=float, default=0.5, 
                        help="fraction of dataset to corrupt for backdoor attack")
    
    parser.add_argument('--robustLR_threshold', type=int, default=0, 
                        help="break ties when votes sum to 0")
    
    
    parser.add_argument('--top_frac', type=int, default=100, 
                        help="compare fraction of signs")
    
    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")

    parser.add_argument('--save_figure', type=str, default='True',
                        help="Figure with poisoned data")
    
    parser.add_argument('--n_figure', type=int, default=3,
                        help="Number of figure with poisoned data")

    parser.add_argument('--class_per_agent', type=int, default=10,
                        help="class per agent for data")
    
    parser.add_argument('--n_classes', type=int, default=10,
                        help="Number of labels in datasets")

    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = args_parser()
    print(args.num_agents)