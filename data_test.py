import re

str = '"劳斯莱斯 寿司 卷'
str = re.sub("\"", " ", str)

print(str.strip())
