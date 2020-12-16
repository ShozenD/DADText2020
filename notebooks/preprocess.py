# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import re
import pandas as pd
from sklearn.model_selection import train_test_split

wiki = pd.read_csv('../data/wikipedia.csv')
uncy = pd.read_csv('../data/uncyclopedia.csv')

wiki.head(10)

uncy.head(10)


# Remove new line character
def parse(x):
    x = re.sub("\n","", x)
    return x


wiki['Field1'] = wiki['Field1'].map(lambda x: parse(x))
uncy['Field1'] = wiki['Field1'].map(lambda x: parse(x))

wiki['label'] = 1
uncy['label'] = 0

data = pd.concat([wiki, uncy])

y, X = data['label'], data['Field1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

train = pd.DataFrame({'label': y_train, 'text': X_train})
test = pd.DataFrame({'label': y_test, 'text': X_test})

test.to_csv('../data/test_orig.txt', sep='\t', index=False, header=False)
train.to_csv('../data/train_orig.txt', sep='\t', index=False, header=False)


