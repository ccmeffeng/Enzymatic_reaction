import random
import numpy as np


data = []
labels = []
seqlen = []

rxns = open('fullset.txt', 'r').readlines()
random.shuffle(rxns)

for line in rxns:
    # Random sequence length
    lst = line.strip().split()
    if len(lst) < 9 or len(lst) > 151:
        continue
    lbl = [0.0] * 7
    lbl[int(lst.pop())] = 1.0
    labels.append(lbl)
    seqlen.append(len(lst))
    for j in range(len(lst)):
        lst[j] = [float(lst[j]) + 1.0]
    data.append(lst)

test_set_label = []
test_set_list = []
training_set_label = []
training_set_list = []
validation_set_label = []
validation_set_list = []

for k in range(len(seqlen)):
    ran = random.random()
    if ran < 0.8:
        training_set_list.append(data[k] + [[0.0]] * (150 - seqlen[k]))
        training_set_label.append(labels[k])
    elif ran < 0.9:
        test_set_list.append(data[k] + [[0.0]] * (150 - seqlen[k]))
        test_set_label.append(labels[k])
    else:
        validation_set_list.append(data[k] + [[0.0]] * (150 - seqlen[k]))
        validation_set_label.append(labels[k])

for ii in range(100):
    # print(validation_set_list[ii])
    print(validation_set_label[ii])

test_set = np.array(test_set_list)
test_set_label = np.array(test_set_label)
print(len(test_set))

training_set = np.array(training_set_list)
training_set_label = np.array(training_set_label)
print(len(training_set))

validation_set = np.array(validation_set_list)
validation_set_label = np.array(validation_set_label)
print(len(validation_set))

np.save('test_set.npy', test_set)
np.save('test_set_label.npy', test_set_label)
np.save('training_set.npy', training_set)
np.save('training_set_label.npy', training_set_label)
np.save('validation_set.npy', validation_set)
np.save('validation_set_label.npy', validation_set_label)
