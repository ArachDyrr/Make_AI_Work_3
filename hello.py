import pandas as pd

# print('hello world')

d = {'col1': ['hello world', 2], 'col2': [3, 4]}

df = pd.DataFrame(data=d)


print(df['col1'][0])