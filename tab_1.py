'''
The code in this module generates the results in table 1. It has to be run 5 times one time per line of the table.
Each run should have different selection of tokens. Please, pay attention to the following:
1. the folder for saving the model 611
2. the folder for saving the visualization data for tensorboard 612
3. The portion of the text that is used as input to the model (96, 435) this two lines should match per run
4. name of the final result file 609,610. here the name of the output file along with the experiment results folder must be specified
according to the model and setting used.
'''

import gc
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef,classification_report, recall_score,precision_score

from transformers import BertTokenizer,BertForSequenceClassification,AdamW, get_linear_schedule_with_warmup

import pandas as pd
import os

from torch.utils.data import TensorDataset, RandomSampler, DataLoader,SequentialSampler

import numpy as np
import random

import time


from torch.utils.tensorboard import SummaryWriter
import utils as my_u

kaggle_data_path = 'data/'  #here is the location of the data resulted from running data_wrangling module

#acc_res = []
#recall_list = []
#fbeta_list = []


def train(output_dir, seed, writer):
    device, gpu_ids = my_u.get_available_devices()

    global acc_res, recall_list, fbeta_list
    acc_res = []
    recall_list = []
    fbeta_list = []

    # Set the seed value all over the place to make this reproducible.
    seed_val = seed

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Load the dataset into a pandas dataframe.
    train_df = pd.read_csv(kaggle_data_path + "train.csv", delimiter='$')

    train_df = train_df[train_df['section_txt'].notna()]

    # Report the number of sentences.
    print('Number of samples (training+validation): {:,}\n'.format(train_df.shape[0]))

    print('Positive Samples: ' + str(len(train_df[train_df['label'] == 1])))
    print('Negative Samples: ' + str(len(train_df[train_df['label'] == 0])))

    # Get the lists of sentences and their labels.
    section_txts = train_df.section_txt.values
    labels = train_df.label.values

    # use tokenizer

    # Load the BERT tokenize:r.
    print('Loading BERT tokenizer for Bert base cased..')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    progress_indicator = 0
    global_indicator = 0

    # For every sentence...
    for txt in section_txts:

        progress_indicator += 1
        if progress_indicator == 10000:
            global_indicator += 10000
            print(global_indicator)
            progress_indicator = 0

        tokens = tokenizer.tokenize(txt)
        # this line changes to consider different parts of the sections
        tokens = tokens[:128] + tokens[-382:]
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
            sent += [0] * (max_len - cur_len)
        assert len(sent) <= max_len, 'required 512, but got {0}'.format(len(sent))
        padded_input_ids.append(sent)

    attention_masks = []

    for sent in padded_input_ids:
        temp_list = [int(token_id > 0) for token_id in sent]
        attention_masks.append(temp_list)
    # -------shuffle the dataset and divide it in train and validation set
    dataset_size = len(labels)
    # generate indexes
    indexes = [*range(dataset_size)]

    # randomize indexes
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

    batch_size = 8 * max(1, len(gpu_ids))

    # train data loader
    trainDataSet = TensorDataset(train_inputs, train_masks, train_labels)
    trainSampler = RandomSampler(trainDataSet)
    trainDataLoader = DataLoader(trainDataSet, batch_size=batch_size, sampler=trainSampler)

    # validation data loader
    validationDataSet = TensorDataset(validation_inputs, validation_masks, validation_labels)
    vallidationSampler = RandomSampler(validationDataSet)
    validationDataLoader = DataLoader(validationDataSet, sampler=vallidationSampler, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,
                                                                output_attentions=False,
                                                                output_hidden_states=False)

    model = nn.DataParallel(model, gpu_ids)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    epochs = 4

    total_num_steps = len(trainDataLoader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_num_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.nan
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(trainDataLoader):

            # Progress update every 5000 batches.
            if step % 5000 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = my_u.format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(trainDataLoader), elapsed))


            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a backward pass
            model.zero_grad()


            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]


            total_loss += torch.sum(loss)  # coss.item()

            # Perform a backward pass to calculate the gradients.
            loss.sum().backward()  # loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # torch.cuda.empty_cache()
            # gc.collect()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(trainDataLoader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss.item())

        print("")
        print("  Average training loss: {0:.3f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(my_u.format_time(time.time() - t0)))
        print('loss so far')
        print(loss_values)
        # tensorboard
        writer.add_scalar("Avg Training Loss", avg_train_loss, epoch_i)

        tokenizer, model = validation(output_dir, model, tokenizer, validationDataLoader, writer, gpu_ids, epoch_i)

    # store training loss and validation acc in a file
    with open(output_dir + 'result.txt', 'w') as file:
        file.write('Avg Training loss\n')
        for item in loss_values:
            file.write("%s\t" % item)
        file.write("\n\n")

        file.write("Avg Validatin Accuracy\n")
        for item in acc_res:
            file.write("%s\t" % item)
        file.write("\n\n")

        file.write("Avg Validation recall: ")
        for item in recall_list:
            file.write("%s\t" % item)
        file.write(" \n\n ")

        file.write("Avg Fbeta score: ")
        for item in fbeta_list:
            file.write("%s\t" % item)
        file.write(" \n\n")

        file.close()

    print("")
    print("Training complete!")


def validation(output_dir, model, tokenizer, validationDataLoader, writer, gpu_ids, epoch_i):
    print("")
    print("Running Validation...")


    t0 = time.time()

    # tensorboard stuff
    tb_labels = []  # store the labels for tensorboard
    tb_preds = []  # store predctions for tensorboard

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy, eval_recall, eval_fbeta = 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validationDataLoader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # tensorboard
        class_preds = [F.softmax(output, dim=0) for output in
                       outputs[0]]  # [F.softmax(outputs[0],dim=1)]#[F.softmax(output,dim=0) for output in outputs]
        labels_flat = label_ids.flatten()
        tb_labels.append(b_labels)
        tb_preds.append(class_preds)

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = my_u.flat_accuracy(logits, label_ids)
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # calculate the recall for this batch of validation sentence
        tmp_eval_recall = my_u.compute_recall(logits, label_ids)
        # accumulate the total recall
        eval_recall += tmp_eval_recall

        # calculatte the recall for this batch of validation examples
        temp_eval_fbeta = my_u.compute_fbeta(logits, label_ids)
        # accumulate the tottal fbeta
        eval_fbeta += temp_eval_fbeta

        # Track the number of batches
        nb_eval_steps += 1


    # tensorboard: add pr_graph
    tb_preds = torch.cat([torch.stack(batch) for batch in tb_preds])
    tb_labels = torch.cat(tb_labels)
    classes = range(2)
    for i in classes:
        labels_i = tb_labels == i
        preds_i = tb_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)

    # Report the final accuracy for this validation run.
    temp_acc = eval_accuracy / nb_eval_steps
    temp_recall = eval_recall / nb_eval_steps
    temp_fbeta = eval_fbeta / nb_eval_steps

    print("  Accuracy: {0:.3f}".format(temp_acc))
    print("  Recall  : {0:.3f}".format(temp_recall))
    print("  Fbeta   : {0:.3f}".format(temp_fbeta))
    print("  Validation took: {:}".format(my_u.format_time(time.time() - t0)))

    if len(acc_res) == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print('first model saved')
    else:
        if temp_acc > float(acc_res[-1]):
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print('Accuracy improved and model saved')
        elif temp_acc < float(acc_res[-1]):
            model = BertForSequenceClassification.from_pretrained(output_dir)
            #model = RobertaForSequenceClassificationWithFL.from_pretrained(output_dir)

            model = nn.DataParallel(model, gpu_ids)
            model.cuda()
            tokenizer = BertTokenizer.from_pretrained(output_dir)
            # tokenizer = RobertaTokenizer.from_pretrained(output_dir)

            print('model reloaded from memory')
    print('Accuracy So Far:')
    acc_res.append(temp_acc)
    recall_list.append(temp_recall)
    fbeta_list.append(temp_fbeta)

    print('AVG Accuracy so far:')
    print(acc_res)
    # tensorboard
    writer.add_scalar("Eval/Accuracy", acc_res[epoch_i], epoch_i)

    print("Recall so far: ")
    print(recall_list)
    writer.add_scalar("Eval/Recall", recall_list[epoch_i], epoch_i)

    print("fbeta so far: ")
    print(fbeta_list)
    writer.add_scalar("Eval/F_beta", fbeta_list[epoch_i], epoch_i)

    writer.close()

    print('validation of epoch ' + str(epoch_i + 1) + ' completed')

    return tokenizer, model

def test(exp_num, output_dir, writer):

    test_data = kaggle_data_path+'test.csv'

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    # Copy the model to the GPU.
    model.cuda()

    # Load the dataset into a pandas dataframe.
    test_df = pd.read_csv(test_data, delimiter='$')

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
        tokens = tokens[:128] + tokens[-382:]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        ids = tokenizer.convert_tokens_to_ids(tokens)
        # Add the encoded text to the list.
        input_ids.append(ids)

    max_len = 512  # max([len(title) for title in input_ids])
    print('Max text length: ', max_len)

    padded_input_ids = []
    for sent in input_ids:
        cur_len = len(sent)
        if cur_len < max_len:
            sent += [0] * (max_len - cur_len)
        sen_len = len(sent)
        assert sen_len <= max_len, f'Required 510 or smaller but got {sen_len}'
        padded_input_ids.append(sent)

    attention_masks = []

    for sent in padded_input_ids:
        temp_list = [int(token_id > 0) for token_id in sent]
        attention_masks.append(temp_list)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    # Set the batch size.
    batch_size = 8 * max(1, len(gpu_ids))

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

    print('Positive samples: %d of %d (%.2f%%)' % (
        test_df.label.sum(), len(test_df.label), (test_df.label.sum() / len(test_df.label) * 100.0)))

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    recall = recall_score(flat_true_labels, flat_predictions)
    prec = precision_score(flat_true_labels, flat_predictions)
    writer.add_scalar("Recall on Testing data", recall, exp_num)
    writer.add_scalar("Precision on Testing data", prec, exp_num)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    writer.add_scalar("MCC On Testing set", mcc, exp_num)

    writer.add_scalars("MCC/Re/Pr on Test data", {'MCC': mcc, 'Re': recall, 'Pr': prec}, exp_num)

    #file = open(output_dir + 'recomputed_result_' + str(exp_num) + '.txt', 'w')
    file = open(output_dir + 'result_' + str(exp_num) + '.txt', 'w')
    repo = classification_report(flat_true_labels, flat_predictions)
    file.write(repo)
    file.write('============================\n')
    file.write('MCC: {}'.format(mcc))

    file.close()
    print()


def main():

    seeds = [42,67,330,2004,945]
    for i in range(len(seeds)):
        print('')
        print('')
        print(f'€€€€€€€€€€€€€€€€€€€€€€€€€€€€ Training Exp {i} €€€€€€€€€€€€€€€€€€€€€€€€€€€€ ')
        seed = seeds[i]
        # output directory to save the trained model

        output_dir = exp_path + str(seed) + "/"
        print(output_dir)

        #create summary writer for tensorboard
        tensorboard_folder = tensor_path+str(i)
        print(tensorboard_folder)
        writer = SummaryWriter(tensorboard_folder)

        #train and evaluate the model
        train(output_dir,seed,writer)

        print("------------------Testing-------------------")
        #test the model
        test(i,output_dir,writer)

        #clean memory
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    device, gpu_ids = my_u.get_available_devices()

    exp_path = "Bert_F128L382/bert__"
    tensor_path = 'saveF128L382/bert_'
    main()
    out_file = "Bert_F128L382/F128L382.txt"
    exp_folder = "./" + exp_path + "/"
    my_u.combine_results(out_file,exp_path)