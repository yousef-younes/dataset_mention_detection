import pandas as pd
import json


project_path = "HERE IS THE PATH THE COLREDIGE DATASET - SHOW US THE DATA"
data_path = project_path + 'data\\'

''''
This function reads kaggle data. extract sections from papers and annotate them with a lable. It also removes short sections (<20) and
also different types of repetitions.
'''

def process_kaggle_data():

    df = pd.read_csv(data_path+'train.csv')

    #create dicctionary of the structure {id:(pub_title,[dataset name, dataset title, dataset lable])}
    dic = {}
    j=0
    pub = set()
    for i, row in df.iterrows():
        id = row[0]
        if id not in dic.keys():
            j+=1
            dic[id]=(row['pub_title'],set())

            if len(row['pub_title']) <5:
                print(row['pub_title'])
            pub.add(row['pub_title'])

        dic[id][1].add(row[1])
        dic[id][1].add(row[2])
        dic[id][1].add(row[3])
    print(len(pub))
    print(j)


    data = []

    for file_id, v in dic.items():

        publication_name = v[0]

        f = open(data_path + "\\train\\" + file_id + '.json')
        text_content = json.load(f)

        # loop through the json pairs
        for section in text_content:
            section_title = section['section_title']
            section_txt = section['text']

            if len(section_txt) < 5:
                print(section_txt)
                continue

            section_txt.replace('$', ' ')

            just_indicat = 0

            positive = False
            for ds in v[1]:
                if section_txt.find(ds) != -1:
                    data_item = (file_id, publication_name, section_title, ds, int(1), section_txt)
                    #print('{},{},{}'.format(file_id, section_title, len(v[1])))
                    positive = True
            if not positive:
                data_item = (file_id, publication_name, section_title, 'None', int(0), section_txt)

            data.append(data_item)

    #create dataframe
    dframe = pd.DataFrame(data, columns=['file_id', 'publication_name', 'section_title', 'dataset', 'label', 'section_txt'])

    print(len(dframe))
    #remove all rows with section_txt less than 20 chars
    dframe = dframe[dframe.section_txt.str.len().gt(20)]
    print('number of samples after removing all sections whose text length is less than 20 chars: {}'.format(len(dframe)))
    #remove all repetitions (file_id, section_txt, dataset, label)
    dframe = dframe[~dframe.duplicated(['file_id','section_txt','dataset','label'],keep='first')]
    print('number of samples after removing all sections whose file_id,section_txt,dataset,label are the same: {}'.format(len(dframe)))

    #remove all repetitions (section_txt, dataset, label)
    dframe = dframe[~dframe.duplicated(['section_txt','dataset','label'],keep='first')]
    print('number of samples after removing all sections whose section_txt,dataset,label are the same: {}'.format(len(dframe)))

    dframe = dframe[~dframe.duplicated(['section_txt','label'],keep='first')]
    print('number of samples after removing all sections whose section_txt,label are the same: {}'.format(len(dframe)))


    #save dataframe to file
    dframe.to_csv(project_path + 'processed_data.csv', index=None, sep=str('$'))

'''
read processed data and divide it into 20% test set and 80% train+dev sets
'''
def divide_processed_data():

    df = pd.read_csv(project_path+'\\processed_data.csv',sep=str('$'))

    test = df.sample(frac=0.2)
    print(len(test))

    train = df.drop(test.index)
    print(len(train))


    test.to_csv(project_path+'\\test.csv',index=None,sep=str('$'))
    train.to_csv(project_path+'\\train.csv',index=None,sep=str('$'))


'''
This function presents some statistics about the test and train sets
'''
def get_stats():

    print("Getting training statstics: ")
    train = pd.read_csv(project_path+'\\train.csv',sep=str('$'))

    all_len = len(train)
    pos_len = len(train[train.label == 1])
    neg_len = len(train[train.label == 0])
    print('number of samples: {}'.format(all_len))
    print('number of positive samples: {}'.format(pos_len))
    print('number of negative samples: {}'.format(neg_len))
    print('percentage of positive samples out of the whole training data {}%'.format(pos_len*100/ all_len))
    print('percentage of positive samples out of the whole training data {}%'.format(neg_len * 100 / all_len))
    train_datasets = set(train.dataset.unique())
    print('number of unique datasets: {}'.format(len(train_datasets)))
    print("datasets are:")
    print(train_datasets)
    print(train.dataset.apply(lambda x: len(x)).value_counts())

    print("*************************************************************************")
    print("Getting testing statstics: ")
    test = pd.read_csv(project_path + '\\test.csv', sep=str('$'))

    all_len = len(test)
    pos_len = len(test[test.label == 1])
    neg_len = len(test[test.label == 0])
    print('number of samples: {}'.format(all_len))
    print('number of positive samples: {}'.format(pos_len))
    print('number of negative samples: {}'.format(neg_len))
    print('percentage of positive samples out of the whole testing data {}%'.format(pos_len * 100 / all_len))
    print('percentage of positive samples out of the whole testing data {}%'.format(neg_len * 100 / all_len))
    test_datasets = set(test.dataset.unique())
    print('number of unique datasets: {}'.format(len(test_datasets)))
    print("datasets are:")
    print(test_datasets)
    print(test.dataset.apply(lambda x: len(x)).value_counts())


    print("**********************************************************************")
    print("shared datasets between train and test:")
    common = test_datasets & train_datasets
    print(len(common))
    print(common)
    print("datasets in train and not in test: ")
    train_dif =  train_datasets - test_datasets
    print(len(train_dif))
    print(train_dif)

    print("datasets in test and not in train: ")
    test_diff = test_datasets - train_datasets
    print(len(test_diff))
    print(test_diff)

get_stats()
