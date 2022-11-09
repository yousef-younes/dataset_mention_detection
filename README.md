# dataset_mention_detection
This is the code for the ICNLSP paper titled "Handling Class Imbalance when Detecting Dataset Mentions with Pre-trained Language Models" which compares the effectiveness of using imbalance handling techniques like resampling and cost sensetive learning with language models like Roberta and BERT.

The code behind each table in the paper is encapsulated in one module whose name is the name of the table e.g., tab_1 module can be used to reproduce the resutls of table 1. At the top of each file there are some experiment specific instructions which indicates the line that need to be changed according to the experiment. After runing each experiment, the average reported results will be printed on screen or saved in a text file which is clear from the code. 

The colredge dataset can be downloaded from https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/data. After that the data_wrangling module can be used to generate the train and test splits provided in the data folder. 

The utils moduls contains code that is used by different experiments. 
