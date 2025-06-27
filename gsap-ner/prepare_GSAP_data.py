#this modules assumes that the GSAP-NER data is in the same directory. It reads the train, validation and test splits from each fold and combine them in a csv file. 
#it also has code to prepare the data for question answering but it is not used here

from datasets import load_from_disk
import pandas as pd
import json
import csv
import pdb


def extract_spans(text, start_indices, end_indices):
    spans = []
    for start, end in zip(start_indices, end_indices):
        span = text[start:end]
        spans.append(span)
    return spans
    
def prepare_data_for_QA(spans):
    # Initialize an empty list to store data items
    data_items = []

    # Loop through your spans or extracted data
    for i, span in enumerate(spans):
        data_item = {
            "id": str(counter_for_id + 1),  # Assuming counter_for_id is defined
            "context": context,  # Replace with your context variable
            "question": "",
            "answers": {
            'text': [span],  # List of extracted spans
            'answer_start': [start_indices[i]]  # List of start indices
           },
           "label": 0,  # Replace with your label value
           "masked_context": context_without_ds_mentions  # Replace with your masked_context variable
           }
        data_items.append(data_item)

    # Optionally, increment counter_for_id
    counter_for_id += len(spans)



def save_csv_file(file_name,data):
    # Write the data to a CSV file using $ as the delimiter
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='$')
        
        # Write each tuple to the CSV file
        for row in data:
            csvwriter.writerow(row)
    
    print(f"Data has been saved to {file_name} successfully.")

def main():
    # Open the JSON file in read mode
    #with open('10-fold-publication-filenames.json', 'r') as file:
        # Load the JSON data into a Python dictionary
        #data = json.load(file)
    gsap_data = []
    gsap_data.append(('id','section_txt','label','dataset'))
    for i in range(10):
        dataset_dict = load_from_disk(f"Paragraph/{i}")
    
        # Accessing individual splits
        train_dataset = dataset_dict['train']
        validation_dataset = dataset_dict['validation']
        test_dataset = dataset_dict['test']

        pos_sample_count = 0 #count number of positive samples
        for sample in train_dataset:
            id = sample['id']
            text = sample['text']
            start = sample['stacked_start']
            end = sample['stacked_end']
            label = sample['stacked_label']
            counter  = 0
            sample_lbl = 0
            #if the paragraph has a dataset mention consider it as positive sample, otherwise negative
            if 3 in label:
                for counter in range(len(label)):
                    if label[counter] == 3:
                        pos_sample_count+=1 
                        sample_lbl = 1
                        dataset_mention = text[start[counter]:end[counter]]
                        gsap_data.append((id,text, sample_lbl,dataset_mention))
                        counter+=1
            else:
                gsap_data.append((id,text, sample_lbl,''))


    #print statistics
    print("Statistics about training data:")
    print(f"number of samples:{len(gsap_data)}")
    print(f"Number o positive samples: {pos_sample_count}")
    print(f"number of negative samples:{len(gsap_data) - pos_sample_count}")
    
    #save data in csv file
    #save_csv_file("../../data/gsap/train.csv", gsap_data)

main()