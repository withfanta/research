import re
a = 'TH.1'
a = a.replace("."," ")
a = re.sub(r'\d+', '', a)
print(a)
