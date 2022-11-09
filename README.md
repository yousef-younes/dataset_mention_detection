# dataset_mention_detection
This is the code for the ICNLSP paper titled "Handling Class Imbalance when Detecting Dataset Mentions with Pre-trained Language Models" which compares the effectiveness of using imbalance handling techniques like resampling and cost-sensitive learning with language models like Roberta and BERT.

The code behind each table in the paper is encapsulated in one module whose name is the table's name, e.g., the tab_1 module can be used to reproduce the results of table 1. At the top of each module are some experiment-specific instructions indicating the lines that need to be changed according to the experiment. After running each experiment, the average reported results will be printed on the screen or saved in a text file that is clear from the code. 

To run the code, please, do the following:

1. Download the Coleridge dataset from https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/data.

2. Use the functions in the data_wrangling module to process the data using process_kaggle_data function. Then split the data into training and testing sets using divide_processed_data function. These splits should reside in a folder named "data" which is in the same directory as the code. A third function also shows statistics about the training and testing splits. Unfortunately, we can not provide the data due to GitHub restrictions on the size. 

3. For the table that you want to reproduce, run its associated module which will show the results when the experiments are done. 
