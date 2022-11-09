import torch
import torch.nn as nn

from transformers import RobertaTokenizer,BertTokenizer,AdamW,get_linear_schedule_with_warmup
from BertClassifierWithFL import BertForSequenceClassificationWithFL
from RobertaClassifierWithFL import RobertaForSequenceClassificationWithFL
import pandas as pd
import os
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler,DataLoader

import numpy as np
import random

import time
import datetime

import gc
from sklearn.metrics import matthews_corrcoef, recall_score, classification_report
import utils as my_u

'''
This module contains the code that produced the results in table 3. It has to be run twice. One with Bert and the other with Roberta.
The required modifications are to comment and uncomment one of two lines according to the model. Here are they:
line 539: Here is the name of the model used in the experiment.
line 97-98: the appropriate tokenizer should be activated and the other should be commented out.
lines 190-191: the appropriate model should be activated and the other should be commented out. 
lines 380-381: also load the appropriate model
lines 386-387: load the appropriate tokenizer
line 396-397: comment and uncomment the appropriate line
lines 400-401: comment and uncomment the appropriate line
'''


kaggle_data_path = 'data/'

seeds=[42,67,330,2004,945]

def main():
    for seed in seeds:
        output_dir = exp_path+'/seed_'+str(seed)

        #train the model
        train(output_dir,seed)
        #test the model
        test(output_dir)
        
        #clern memory
        torch.cuda.empty_cache()
        gc.collect()
        print('*************************************************************')

    print('All experiments are done')

def train(output_dir,seed):
    acc_res = []
    
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = seed

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Load the dataset into a pandas dataframe.
    train_df = pd.read_csv(kaggle_data_path+"train.csv",delimiter='$')
    train_df=train_df[train_df['section_txt'].notna()]

    #duplicate positive samples and shuffle
    pos_df = train_df[train_df['label']==1]
    neg_df = train_df[train_df['label']==0]
    print('Neg: {},Pos: {}\n'.format(len(neg_df),len(pos_df)))

    neg_df = neg_df.sample(frac=0.55,random_state=seed_val)
    frames = [pos_df,pos_df,neg_df,pos_df,pos_df]
    train_df = pd.concat(frames)
    train_df = train_df.sample(frac=1,random_state=seed_val)

    # Report the number of sentences.
    print('Number of samples (training+validation): {:,}\n'.format(train_df.shape[0]))

    print('Positive Samples: '+str(len(train_df[train_df['label']==1])))
    print('Negative Samples: '+str(len(train_df[train_df['label']==0])))


    # Get the lists of sentences and their labels.
    section_txts = train_df.section_txt.values
    labels = train_df.label.values

    # Load the BERT tokenize:r.
    print('Loading Roberta tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    progress_indicator = 0
    global_indicator = 0

    # For every sentence...
    for txt in section_txts:

        progress_indicator+=1
        if progress_indicator == 10000 :
            global_indicator +=10000
            print(global_indicator)
            progress_indicator=0

        tokens = tokenizer.tokenize(txt)
        tokens = tokens[:382]+tokens[-128:]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        ids = tokenizer.convert_tokens_to_ids(tokens)
        # Add the encoded text to the list
        
        input_ids.append(ids)


    max_len = 512
    print('Max text length: ', max_len)

    padded_input_ids = []
    for sent in input_ids:
        cur_len = len(sent)
        if cur_len < max_len:
            sent += [0]*(max_len-cur_len)
        assert len(sent) <= max_len, 'required 512, but got {0}'.format(len(sent))
        padded_input_ids.append(sent)


    attention_masks = []

    for sent in padded_input_ids:

        temp_list = [int(token_id>0) for token_id in sent]
        attention_masks.append(temp_list)
    #-------shuffle the dataset and divide it in train and validation set
    dataset_size = len(labels)
    #generate indexes
    indexes = [*range(dataset_size)]

    #randomize indexes
    random.shuffle(indexes)

    val_size = int(dataset_size * 0.1)
    train_size = int(dataset_size - val_size)

    train_indexes = indexes[:train_size]
    val_indexes = indexes[-val_size:]
    train_txt = [padded_input_ids[idx] for idx in train_indexes]
    train_labels = [labels[idx] for idx in train_indexes]

    val_txt = [padded_input_ids[idx] for idx in val_indexes]
    val_labels = [labels[idx] for idx in val_indexes]

    train_masks = [attention_masks[idx] for idx in train_indexes]
    val_masks = [attention_masks[idx] for idx in val_indexes]

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_txt)
    validation_inputs = torch.tensor(val_txt)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(val_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(val_masks)


    batch_size =8*max(1,len(gpu_ids))

    #train data loader
    trainDataSet = TensorDataset(train_inputs,train_masks,train_labels)
    trainSampler = RandomSampler(trainDataSet)
    trainDataLoader = DataLoader(trainDataSet,batch_size=batch_size,sampler=trainSampler)

    #validation data loader
    validationDataSet = TensorDataset(validation_inputs,validation_masks,validation_labels)
    vallidationSampler = RandomSampler(validationDataSet)
    validationDataLoader = DataLoader(validationDataSet,sampler=vallidationSampler,batch_size=batch_size)


    model = RobertaForSequenceClassificationWithFL.from_pretrained("roberta-base",num_labels=2,output_attentions=False,output_hidden_states=False)
    #model =BertForSequenceClassificationWithFL.from_pretrained("bert-base-uncased",num_labels=2,output_attentions=False,output_hidden_states=False)
    model.gamma =4
    model.alpha = 0.12
    model = nn.DataParallel(model,gpu_ids)
    model.cuda()
    
    optimizer = AdamW(model.parameters(),lr=2e-5,eps=1e-8)

    epochs = 4

    total_num_steps = len(trainDataLoader)*epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_num_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    #recall 
    #eval_recall = [0,0]

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.nan
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(trainDataLoader):

            # Progress update every 5000 batches.
            if step % 5000 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = my_u.format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(trainDataLoader), elapsed))


            # unpack the batch and move tensors to gpu memory

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()


            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += torch.sum(loss) #coss.item()

            # Perform a backward pass to calculate the gradients.
            loss.sum().backward()#loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            
            #torch.cuda.empty_cache()
            #gc.collect()       
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(trainDataLoader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss.item())

        print("")
        print("  Average training loss: {0:.3f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(my_u.format_time(time.time() - t0)))
        print('loss so far')
        print(loss_values)
        
           
    # ========================================
        #               Validation
        # ========================================
        # After the completion of each 2000 training steps, measure our performance on
        # our validation set.
            
        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validationDataLoader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = my_u.flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
        #print('recall values')


        # Report the final accuracy for this validation run.
        temp_acc = eval_accuracy/nb_eval_steps
        print("  Accuracy: {0:.3f}".format(temp_acc))
        print("  Validation took: {:}".format(my_u.format_time(time.time() - t0)))
        if len(acc_res) == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print('first model saved')
        else:
            if temp_acc > float(acc_res[-1]):
                model_to_save = model.module if hasattr(model,'module') else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print('Accuracy improved and model saved')
            elif temp_acc < float(acc_res[-1]):
                #model = BertForSequenceClassificationWithFL.from_pretrained(output_dir)
                model = RobertaForSequenceClassificationWithFL.from_pretrained(output_dir)
                model.gamma =4
                model.alpha = 0.12
                model = nn.DataParallel(model,gpu_ids)
                model.cuda()
                tokenizer = RobertaTokenizer.from_pretrained(output_dir)
                #tokenizer = BertTokenizer.from_pretrained(output_dir)
                
                print('model reloaded from memory')
        print('Accuracy So Far:')
        acc_res.append(temp_acc)
        print(acc_res)

def test(output_dir):
    #load model
    model =RobertaForSequenceClassificationWithFL.from_pretrained(output_dir)
    #model = BertForSequenceClassificationWithFL.from_pretrained(output_dir)
    model.gamma = 4
    model.alpha = 0.12
    tokenizer = RobertaTokenizer.from_pretrained(output_dir)
    #tokenizer = BertTokenizer.from_pretrained(output_dir)

    # Copy the model to the GPU.
    model.cuda()


    # Load the dataset into a pandas dataframe.
    test_df = pd.read_csv(kaggle_data_path+"test.csv", delimiter='$')
    #test_df = test_df.head(10)
    test_df = test_df[test_df['section_txt'].notna()]


    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(test_df.shape[0]))

    # Create sentence and label lists
    section_txts = test_df.section_txt.values
    labels = test_df.label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []


    progress_indicator = 0
    global_indicator = 0

    # For every sentence...
    for txt in section_txts:

        progress_indicator += 1
        if progress_indicator == 1000:
            global_indicator += 1000
            print(global_indicator)
            progress_indicator = 0

        tokens = tokenizer.tokenize(txt)
        tokens = tokens[:328]+tokens[-128:]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        ids = tokenizer.convert_tokens_to_ids(tokens)
        # Add the encoded text to the list.
        input_ids.append(ids)

    max_len = max([len(title) for title in input_ids])
    print('Max text length: ', max_len)

    padded_input_ids = []
    for sent in input_ids:
        cur_len = len(sent)
        if cur_len < max_len:
            sent += [0]*(max_len-cur_len)
        sen_len = len(sent)
        assert sen_len <= max_len, f'Required 510 or smaller but got {sen_len}'
        padded_input_ids.append(sent)

    attention_masks = []

    for sent in padded_input_ids:

        temp_list = [int(token_id>0) for token_id in sent]
        attention_masks.append(temp_list)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    # Set the batch size.
    batch_size = 8*max(1,len(gpu_ids))

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    print('Positive samples: %d of %d (%.2f%%)' % (test_df.label.sum(), len(test_df.label), (test_df.label.sum() / len(test_df.label) * 100.0)))


    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    
    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    file = open(exp_path+'/result.txt','a')

    file.write('MCC: %.3f\n' % mcc)

    file.write(classification_report(flat_true_labels,flat_predictions))
    file.write('\n********************************\n\n')
    
    file.close()

if __name__ == '__main__':
    exp_path =  'Roberta' #'Bert'
    device, gpu_ids = my_u.get_available_devices()
    main()
    my_u.avg_calculation(exp_path)



