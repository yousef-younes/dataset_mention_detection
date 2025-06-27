
seeds = [42,67,330,2004,945]

f = open("roberta_content_results.txt", "w")

f.write("Results on test set\n")

#collect the classification reports in one file
for i in range(len(seeds)):

    f.write('EXP {0} \n'.format(i))
    file_path= './Roberta_gsap/roberta_'+str(seeds[i])+'/result' 

    temp_file =open(file_path + '_'+str(i)+'.txt',"r")

    for line in temp_file:
        f.write(line)
    temp_file.close()

    temp_file = open(file_path+".txt", 'r')
    for line in temp_file:
        f.write(line)
    temp_file.close()

    f.write('\n\n\n')

    f.write('----------------------------------------------------------------')
    f.write('\n\n') 



f.write('The average results to be reported\n')

n_prec = n_recall = n_f1_score = 0.0
p_prec = p_recall = p_f1_score = 0.0
acc=0.0
mcc = 0.0

#compute the average results
for i in range(5):

    
    file_path= './Roberta_gsap/roberta_'+str(seeds[i])+'/result_'+str(i)+'.txt'
    
    temp_file = open(file_path,'r')
    
    line_number = 0

    for line in temp_file:
        if line_number == 2:
            n_prec +=float(line[19:23])
            n_recall +=float(line[29:33])
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

        line_number +=1

print(n_prec)
print(acc)

f.write('negative class:\n')
f.write('precesion: {}, recall: {}, f1_score: {}\n'.format(n_prec/5,n_recall/5,n_f1_score/5))
f.write('positive class:\n')
f.write('precesion: {}, recall: {}, f1_score: {}\n'.format(p_prec/5,p_recall/5,p_f1_score/5))
f.write('Accuracy: {}\n'.format(str(acc/5)))
f.write('MCC : {}'.format(str(mcc/5)))


    
f.close()




