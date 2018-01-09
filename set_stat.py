
data = open('fullset.txt', 'r').readlines()
stat = [0] * 463
for line in data:
    l = len(line.strip().split()) - 1
    stat[l] += 1

for i in range(len(stat)):
    if stat[i] != 0:
        print('%d %d'%(i, stat[i]))
