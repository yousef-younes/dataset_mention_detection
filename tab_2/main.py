
import gc
import torch.cuda

from train import train
from test import test

from torch.utils.tensorboard import SummaryWriter

def main():

    seeds = [42,67,330,2004,945]
    for i in range(5):
        print('')
        print('')
        print(f'€€€€€€€€€€€€€€€€€€€€€€€€€€€€ Training Exp {i} €€€€€€€€€€€€€€€€€€€€€€€€€€€€ ')
        seed = seeds[i]
        # output directory to save the trained model
        output_dir = "Robert/exp__" + str(seed) + "/"
        print(output_dir)
        #create summary writer for tensorboard
        tensorboard_file = 'save_robert/exp_'+str(i)
        print(tensorboard_file)
        writer = SummaryWriter(tensorboard_file)

        #train and evaluate the model
        train(output_dir,seed,writer)

        print("------------------Testing-------------------")
        #test the model
        test(i,output_dir,writer)

        #clean memory
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
