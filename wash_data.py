import urllib.request
from rdkit.Chem import AllChem as Chem
import re

num_2_smiles = {}
cmp_count = {}
rxn_data = []


def smiles(s):
    if s not in cmp_count:
        cmp_count[s] = 1
        # 网址
        url = "http://www.genome.jp/dbget-bin/www_bget?-f+m+compound+%s"%s
        # 请求
        request = urllib.request.Request(url)
        # 爬取结果
        response = urllib.request.urlopen(request)
        data = response.read()
        # print(data)
        fl = open('01.mol', 'wb+')
        fl.write(data)
        fl.close()

        moll = Chem.MolFromMolFile('01.mol')
        if moll:
            m = Chem.MolToSmiles(moll)
        else:
            m = 'None'
        num_2_smiles[s] = m
    else:
        cmp_count[s] += 1
        m = num_2_smiles[s]
    return m


lines = open('reactions_unranked.txt', 'r', encoding='gbk').readlines()
for i in range(3000):
    line = lines[i]
    spr = re.sub('\|', ' ', line).strip().split()
    # print(spr)
    if len(spr) == 1:
        continue
    lst1, lst2 = [], []
    for st in spr:
        if st[0] == 'C':
            lst2.append(smiles(st))
            lst1.append(st)
        elif st == '=':
            lst2.append(st)
            lst1.append(st)
    lst1 += ['xx'] + lst2 + [spr[0]]
    if not re.search('\d\.\d+\..+?\..+', line):
        dta = ['0.0.0.0']
    else:
        dta = re.findall('\d\.\d+\.\d+\.\d+|\d\.\d+\.\d+\.-', line)
    for dt in dta:
        new_line = dt.split('.')
        for nm in new_line:
            if re.search('\d',nm):
                nm = int(nm)
        new_line += lst1
        print(new_line)
        rxn_data.append(new_line)

# 把化合物出现频率的词典存储下来
cmp_count_lst = []
compounds_freq = open('cmp_freq.txt', 'w', encoding='gbk')
for k, v in cmp_count.items():
    cmp_count_lst.append((v, k))
cmp_count_lst.sort(reverse=True)
for cmp in cmp_count_lst:
    compounds_freq.write(str(cmp[0]) + ' ' + str(cmp[1]) + '\n')
compounds_freq.close()

# 把编号到SMILES的词典存储下来
n2s = open('num2smiles.txt', 'w', encoding='gbk')
for kk, vv in num_2_smiles.items():
    n2s.write(str(kk) + ' ' + vv + '\n')
n2s.close()

# 把最终的表格存下来
rxn_data.sort()
rxn = open('rxn_data.txt', 'w', encoding='gbk')
for dtline in rxn_data:
    for elm in dtline:
        rxn.write(str(elm) + ' ')
    rxn.write('\n')
rxn.close()
