import urllib.request
from rdkit.Chem import AllChem as Chem

# 反应一共到11896

# 网址
url = "http://www.genome.jp/dbget-bin/www_bget?-f+m+compound+C00404"
# 请求
request = urllib.request.Request(url)
# 爬取结果
response = urllib.request.urlopen(request)
data = response.read()

fl = open('01.mol', 'wb+')
fl.write(data)
fl.close()

m = Chem.MolToSmiles(Chem.MolFromMolFile('01.mol'))
x = Chem.MolToSmiles(Chem.MolFromMolFile('01.mol'))

print(m)
