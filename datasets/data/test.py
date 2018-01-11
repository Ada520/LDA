# -*- encoding:utf-8 -*-

import os
import re
import codecs
import nltk

SEPS = '[\s()-/,:.?!\>\<\\\\\^\-\"\=]\s*'


stopwords = set([])# #共127个
stopwords.update(['from', 'subject', 'writes','|','[',']','cc',''])
f=codecs.open('test.txt','w','utf-8')
for file in os.listdir('raw/')[6:10]:
    if file!='metadata':
        for text in os.listdir('raw/'+file)[:15]:
            content=codecs.open('raw/'+file+'/'+text,'r','utf-8').read()
            texts=re.split(SEPS,content.lower())
            while '' in texts:
                texts.remove('')
            texts=[t for t in texts if t not in stopwords]
            t=" ".join(texts)
            f.write(t+'\n')
f.close()

