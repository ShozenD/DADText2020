# -*- coding: utf-8 -*-
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
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

yahoo = pd.concat([pd.read_csv(f,index_col=False) for f in glob('../data/yahoo*')])

yahoo = yahoo.reset_index(drop=True)
yahoo.tail(10)

# +
# Remove new line character
to_hankaku = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}

def parse(x):
    x = re.sub('[『』〝〟【】「」“”．《》＜＞\[\]〔〕＝]','',x) #『』を除去
    x = re.sub('[&＆・×]','と', x)
    x = re.sub('〜','から',x)
    x = re.sub('　','',x) # 無駄な空白を除去
    x = x.translate(str.maketrans(to_hankaku)) # 全角→半角変換
    x = x.lower() # 大文字→小文字
    
    x = re.sub('[0-9]+','0',x) # 番号を全て0にする
    x = re.sub('[!?]','。',x)
    x = re.sub('…','。',x)
    x = re.sub('。+','。',x)
    
    return x


# -

yahoo['Title'] = yahoo['Title'].map(lambda x: parse(x)).values

yahoo.head(50)

yahoo = yahoo.loc[:,['label','Title']]
yahoo.drop_duplicates()

y, X = yahoo['label'], yahoo['Title']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

train = pd.DataFrame({'label': y_train, 'text': X_train})
test = pd.DataFrame({'label': y_test, 'text': X_test})

len(train)

len(test)

test.to_csv('../data/test.txt', sep='\t', index=False, header=False)
train.to_csv('../data/train.txt', sep='\t', index=False, header=False)

test.label.mean()

train.label.mean()
