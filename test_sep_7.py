import random
import numpy as np


data = []
labels = []

rxns = open('fullset.txt', 'r').readlines()

training_count = [0] * 7
test_count = [0] * 7
validation_count = [0] * 7

for line in rxns:
    # Random sequence length
    lst = line.strip().split()
    lbl = [0.0] * 7
    y = int(lst.pop())
    lbl[y] = 1.0
    labels.append(lbl)
    ln = len(lst)
    new_lst = []
    for j in range(150):
        one_hot = [0.0] * 41
        if j < ln:
            # print(lst[j])
            x = int(lst[j])
            one_hot[x] = 1.0
        new_lst.append(one_hot)
    lst = new_lst[::-1]
    data.append(lst)

test_set_label = []
test_set_list = []
training_set_label = []
training_set_list = []
validation_set_label = []
validation_set_list = []

for k in range(len(data)):
    ran = random.random()
    if ran < 0.8:
        training_set_list.append(data[k])
        training_set_label.append(labels[k])
        for i in range(7):
            if labels[k][i] == 1.0:
                training_count[i] += 1
    elif ran < 0.9:
        test_set_list.append(data[k])
        test_set_label.append(labels[k])
        for i in range(7):
            if labels[k][i] == 1.0:
                test_count[i] += 1
    else:
        validation_set_list.append(data[k])
        validation_set_label.append(labels[k])
        for i in range(7):
            if labels[k][i] == 1.0:
                validation_count[i] += 1

for ii in range(100):
    print(validation_set_list[ii])
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

print(training_count)
print(test_count)
print(validation_count)

np.save('test_set_o.npy', test_set)
np.save('test_set_label_o.npy', test_set_label)
np.save('training_set_o.npy', training_set)
np.save('training_set_label_o.npy', training_set_label)
np.save('validation_set_o.npy', validation_set)
np.save('validation_set_label_o.npy', validation_set_label)
