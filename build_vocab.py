import random

f = open('fullset_3.txt', 'r').readlines()

random.shuffle(f)
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')
f3 = open('dev.txt', 'w')

for elm in f:
    x = random.random()
    if x < 0.8:
        f1.write(elm)
    elif x < 0.9:
        f2.write(elm)
    else:
        f3.write(elm)

f1.close()
f2.close()
f3.close()

