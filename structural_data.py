import re

num_2_smiles = {}
fl_1 = open('num2smiles.txt', 'r').readlines()
for ln1 in fl_1:
    lst1 = ln1.strip().split()
    num_2_smiles[lst1[0]] = lst1[1]

cmp_count = {}
fl_2 = open('cmp_freq.txt', 'r').readlines()
for ln2 in fl_2:
    lst2 = ln2.strip().split()
    cmp_count[lst2[1]] = int(lst2[0])

pre_train_data = []

rxns = open('rxn_data.txt', 'r').readlines()
for rxn in rxns:
    lst = rxn.strip().split()
    get_dta = 0
    reactant = []
    product = []
    for i in range(len(lst)):
        if lst[i] == '=':
            get_dta = 1
        elif lst[i] == 'xx':
            break
        elif i > 3 and get_dta == 0 and num_2_smiles[lst[i]] != 'None':
            # print(lst[i])
            reactant.append((cmp_count[lst[i]], num_2_smiles[lst[i]]))
        elif get_dta == 1 and num_2_smiles[lst[i]] != 'None':
            product.append((cmp_count[lst[i]], num_2_smiles[lst[i]]))
    print('%d %d' % (len(reactant), len(product)))
    if len(reactant) == 0 or len(product) == 0:
        continue
    else:
        reactant.sort()
        product.sort()
        lst2 = lst[0:3] + [reactant[0][1]] + [product[0][1]] + [lst[-1]]
        pre_train_data.append(lst2)

out_file = open('pre_tran_data_2.txt', 'w')
for line in pre_train_data:
    for i in line:
        out_file.write(str(i) + ' ')
    out_file.write('\n')

