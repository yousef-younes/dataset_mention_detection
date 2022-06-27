import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,SequentialSampler,DataLoader

from transformers import RobertaTokenizer,BertTokenizer
import sys
sys.path.append('../')
from RobertaClassifierWithFL import RobertaForSequenceClassificationWithFL
from BertClassifierWithFL import BertForSequenceClassificationWithFL

from sklearn.metrics import matthews_corrcoef,classification_report, recall_score,precision_score

import numpy as np

# If there's a GPU available...
def get_available_devices():
    gpu_ids=[]
    
    if torch.cuda.is_available():
        gpu_ids +=[gpu_id for gpu_id in range(torch.cuda.device_count())]
        # Tell PyTorch to use the GPU.
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

    return device, gpu_ids

test_data = '~/mod_sec_work/data/test.csv'

def test(exp_num,output_dir,writer):
    device, gpu_ids = get_available_devices()

    # Load a trained model and vocabulary that you have fine-tuned
    model = RobertaForSequenceClassificationWithFL.from_pretrained(output_dir)
    #model = BertForSequenceClassificationWithFL.from_pretrained(output_dir)
    model.gamma = 4
    model.alpha = 0.12
    tokenizer = RobertaTokenizer.from_pretrained(output_dir)
    #tokenizer = BertTokenizer.from_pretrained(output_dir)
    # model = nn.DataParallel(model,gpu_ids)
    # Copy the model to the GPU.
    model.cuda()

    # Load the dataset into a pandas dataframe.
    test_df = pd.read_csv(test_data, delimiter='$')

    test_df = test_df[test_df['section_txt'].notna()]
    #test_df = test_df[test_df['section_title'].str.len().lt(100)]
    #test_df = test_df.head(100)
    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(test_df.shape[0]))

    # Create sentence and label lists
    section_txts = test_df.section_txt.values
    labels = test_df.label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    progress_indicator = 0
    global_indicator = 0

    excluded_titles = 0

    # For every sentence...
    for txt in section_txts:

        progress_indicator += 1
        if progress_indicator == 1000:
            global_indicator += 1000
            print(global_indicator)
            progress_indicator = 0

        tokens = tokenizer.tokenize(txt)
        tokens = tokens[:328] + tokens[-128:]
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

    recall = recall_score(flat_true_labels,flat_predictions)
    prec= precision_score(flat_true_labels,flat_predictions)
    writer.add_scalar("Recall on Testing data",recall,exp_num)
    writer.add_scalar("Precision on Testing data",prec,exp_num)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    writer.add_scalar("MCC On Testing set", mcc, exp_num)

    writer.add_scalars("MCC/Re/Pr on Test data",{'MCC': mcc,'Re':recall,'Pr':prec},exp_num)

    print('MCC: %.3f' % mcc)

    #print(classification_report(flat_true_labels, flat_predictions))
    file = open(output_dir+'result_'+str(exp_num)+'.txt','w')
    repo = classification_report(flat_true_labels, flat_predictions)
    file.write(repo)
    file.write('=========================\n')
    file.write('MCC: {}'.format(mcc))

    file.close()
    print()


