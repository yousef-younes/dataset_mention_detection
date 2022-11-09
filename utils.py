''''
This file contains utility functions that are used by different experiments.
'''
import torch
import datetime
import  numpy as np
from sklearn.metrics import recall_score, fbeta_score

'''

this function takes the name of the file to create, and the path of the experiment on which to combine the results of an expriment 
e.g., out_file = exp_F128L382.txt
      exp_path = './Bert_F128L382/bert__'

This function is used by table 1 and 2

'''

def combine_results(out_file, exp_path):
    seeds = [42, 67, 330, 2004, 945]

    f = open(out_file, "w")

    f.write("Results on test set\n")

    # collect the classification reports in one file
    for i in range(len(seeds)):

        f.write('EXP {0} \n'.format(i))
        file_path = exp_path + str(seeds[i]) + '/result'

        temp_file = open(file_path + '_' + str(i) + '.txt', "r")

        for line in temp_file:
            f.write(line)
        temp_file.close()

        temp_file = open(file_path + ".txt", 'r')
        for line in temp_file:
            f.write(line)
        temp_file.close()

        f.write('\n\n\n')

        f.write('----------------------------------------------------------------')
        f.write('\n\n')

    n_prec = n_recall = n_f1_score = 0.0
    p_prec = p_recall = p_f1_score = 0.0
    acc = 0.0
    mcc = 0.0

    # compute the average results
    for i in range(len(seeds)):

        file_path = exp_path + str(seeds[i]) + '/result_' + str(i) + '.txt'

        temp_file = open(file_path, 'r')

        line_number = 0

        for line in temp_file:
            if line_number == 2:
                n_prec += float(line[19:23])
                n_recall += float(line[29:33])
                n_f1_score += float(line[39:43])
            elif line_number == 3:
                p_prec += float(line[19:23])
                p_recall += float(line[29:33])
                p_f1_score += float(line[39:43])
            elif line_number == 5:
                acc += float(line[39:44])
            elif line_number == 9:
                cur_mcc = str(line[5:].strip())
                mcc += float(cur_mcc)

            line_number += 1

    print(n_prec)
    print(acc)

    f.write('The average results to be reported\n')
    f.write('negative class:\n')
    f.write('precesion: {}, recall: {}, f1_score: {}\n'.format(n_prec / 5, n_recall / 5, n_f1_score / 5))
    f.write('positive class:\n')
    f.write('precesion: {}, recall: {}, f1_score: {}\n'.format(p_prec / 5, p_recall / 5, p_f1_score / 5))
    f.write('Accuracy: {}\n'.format(str(acc / 5)))
    f.write('MCC : {}'.format(str(mcc / 5)))

    f.close()

'''
This function is used by table 3 to compute the average of all experiments. It takes one argument which is the name of the folder in
which the experiments results are stored.
'''

def avg_calculation(folder):
    file = open(folder + '/result.txt', 'a+')

    n_prec = n_recall = n_f1_score = 0.0
    p_prec = p_recall = p_f1_score = 0.0
    acc = 0.0
    mcc = 0.0

    line_num = 0
    counter = 5
    for line in file:
        if line_num % 12 == 0:
            print('***************************')
            line_num = 0
            counter -= 1
        if counter < 0:
            break
        if line_num == 0:
            print('MCC: {},{}'.format(line.strip(), line_num))
            mcc += float(line[5:].strip())
        elif line_num == 3:
            print("N: {},{}".format(line.strip(), line_num))
            n_prec += float(line[19:23])
            n_recall += float(line[29:33])
            n_f1_score += float(line[39:43])
        elif line_num == 4:
            print('P:  {},{}'.format(line.strip(), line_num))
            p_prec += float(line[19:23])
            p_recall += float(line[29:33])
            p_f1_score += float(line[39:43])
        elif line_num == 6:
            print('Acc:  {},{}'.format(line.strip(), line_num))
            acc += float(line[39:43])
        line_num += 1

    file.write('******************Final Results**********************\n')
    file.write('N: {},{},{}\n'.format(n_prec / 5, n_recall / 5, n_f1_score / 5))
    file.write('P: {},{},{}\n'.format(p_prec / 5, p_recall / 5, p_f1_score / 5))
    file.write('Acc:{}\n'.format(str(acc/5)))
    file.write('MCC: {}\n'.format(str(mcc/5)))


    file.close()

def get_available_devices():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# function to calculate the recall
def compute_recall(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, pred_flat)
#function to calculate the f_beta score
def compute_fbeta(preds,labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return fbeta_score(labels_flat, labels_flat,beta=3)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
