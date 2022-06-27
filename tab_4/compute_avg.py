def avg_calculation(folder):

    file  = open(folder+'/result.txt','r')
    
    n_prec = n_recall = n_f1_score = 0.0
    p_prec = p_recall = p_f1_score = 0.0
    acc =0.0
    mcc =0.0
   
    line_num = 0
    counter= 5
    for line in file:
        if line_num % 12 == 0:
            print('***************************')
            line_num = 0
            counter-=1
        if counter < 0:
            break;
        if line_num == 0:
            print('MCC: {},{}'.format(line.strip(),line_num))
            mcc += float(line[5:].strip())
        elif line_num == 3:
            print("N: {},{}".format(line.strip(),line_num))
            n_prec += float(line[19:23])
            n_recall += float(line[29:33])
            n_f1_score += float(line[39:43])
        elif line_num ==4:
            print('P:  {},{}'.format(line.strip(),line_num))   
            p_prec += float(line[19:23])
            p_recall +=float(line[29:33])
            p_f1_score += float(line[39:43])
        elif line_num ==6:
            print('Acc:  {},{}'.format(line.strip(),line_num))
            acc +=float(line[39:43])
        line_num +=1
    print('******************Final Results**********************')
    print('N: {},{},{}'.format(n_prec/5,n_recall/5,n_f1_score/5))
    print('P: {},{},{}'.format(p_prec/5,p_recall/5,p_f1_score/5))
    print('Acc: '+str(acc/5))
    print('MCC: '+str(mcc/5))

avg_calculation('Roberta')
