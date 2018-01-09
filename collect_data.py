import urllib.request
from rdkit.Chem import AllChem as Chem
import re

# 反应一共到11901
# http://www.kegg.jp/dbget-bin/www_bget?rn:R11901


def get_equ(s):
    lst = re.findall('C\d{5}</a>|=>',s)
    rst = ''
    for wrd in lst:
        rst += re.sub('<|>|/a','',wrd) + ' '
    return rst
def get_enz(s):
    wrds = re.findall('>\d\.\d+\..+?\..+?<', s)
    rst = ''
    for wrd in wrds:
        rst += re.sub('<|>', '', wrd) + ' '
    return rst

fl = open('reactions.txt','w',encoding='gbk')

# 网址
for i in range(5104, 12500):
    s0 = str(i)
    s = (5-len(s0))*'0' + s0
    url = "http://www.kegg.jp/dbget-bin/www_bget?rn:R%s"%s
    # 请求
    request = urllib.request.Request(url)
    # 爬取结果
    response = urllib.request.urlopen(request)
    data = response.read()
    lines = str(data).split('\\n')
    # 从135行开始是有用的信息
    for j in range(135, len(lines)):
        if re.search('<nobr>Equation</nobr>', lines[j]):
            s += '||' + get_equ(lines[j+1]) + '||'
        elif re.search('<nobr>Enzyme</nobr>', lines[j]):
            s += get_enz(lines[j+1])
            break
    print(s)
    fl.write(s+'\n')

fl.close()
