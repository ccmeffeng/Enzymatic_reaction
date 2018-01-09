import random

dta = open('pre_tran_data_2.txt', 'r').readlines()

training_set = []
testing_set = []
smiles2num = {}


def s2n(sm):
    num_list = []
    for ch in sm:
        if ch not in smiles2num:
            leng = len(smiles2num)
            smiles2num[ch] = leng
            num_list.append(leng)
        else:
            num_list.append(smiles2num[ch])
    return num_list

f = open('fullset_2.txt', 'w')
max_len = 0
for line in dta:
    lst = line.strip().split()
    l = s2n(lst[3] + '.') + s2n(lst[4])
    # print(s2n(lst[3] + '.'))
    if len(l) < 15 or len(l) > 300:
        continue
    if len(l) > max_len:
        max_len = len(l)
    for i in l:
        f.write(str(i) + ' ')
    f.write(str(lst[0]) + '\n')

print(max_len)
