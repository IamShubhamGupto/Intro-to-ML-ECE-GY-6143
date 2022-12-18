import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = [
    ['chub',	1,	0,	0,	0,	1,	'fish'],
    ['seahorse',	0,	0,	0,	0,	1,	'fish'],
	['skimmer',	1,	0,	0,	1,	0,	'bird'],
	['stingray',	1,	0,	0,	0,	1,	'fish'],
	['flamingo', 0,	0,	0,	1,	0,	'bird'],
	['gull',	1,	0,	0,	1,	0,	'bird'],
	['vulture',	1,	0,	0,	1,	0,	'bird'],
	['parakeet',	0,	0,	1,	1,	0,	'bird'],
	['dogfish',	1,	0,	0,	0,	1,	'fish'],
	['herring',	1,	0,	0,	0,	1,	'fish']
    ]   

columns = ['name',	'predator',	'milk',	'domestic',	'feathers',	'toothed',	'category']
en_columns = columns[1:-1]
df = pd.DataFrame(data, columns=columns)

labelencoder = LabelEncoder()
df['category_cat'] = labelencoder.fit_transform(df['category'])
print(df)
target_column_count = dict(df['category_cat'].value_counts())
total_rows = sum(list(target_column_count.values()))
entropy = 0
for count in list(target_column_count.values()):
    entropy = entropy - (count/total_rows)*np.log(count/total_rows)/np.log(2)

print(f"entropy = {entropy}")

for column in en_columns:
    en_column_map = dict(df[column].value_counts())
    g_sum = df.groupby(column)['category_cat'].value_counts().to_frame('count')
    g_sum.reset_index(inplace=True)
    en_col_map = dict()
    cat_entropy = 0
    for en_column_cat in en_column_map.keys():
        en_entropy = 0
        for target_column_cat in target_column_count.keys():
            try:
                count = g_sum[(g_sum[column]==en_column_cat) & (g_sum['category_cat']==target_column_cat)]['count'].values[0]
            except:
                continue
            # print("count = ",count, type(count))
            en_entropy = en_entropy - (count/en_column_map[en_column_cat])*np.log(count/en_column_map[en_column_cat])/np.log(2)
    
        if(en_entropy != 0.0):
            cat_entropy += (en_column_map[en_column_cat]/total_rows)*en_entropy
    print(f"Entropy(S|{column}) = {cat_entropy}")
    print(f"Gain = {entropy - cat_entropy}")
    
    