# This is a simple test to know the gensim about lda part easily

import jieba
import os
from gensim import corpora,models,similarities
train_set=[]
files=open('news.txt',encoding='utf-8').readlines()

for raw in files:
    word_list=raw.strip().split()
    train_set.append(word_list)

dic=corpora.Dictionary(train_set)#词典
# for k,v in dic.items():
#     print(k,v)#（词序号，词汇）
##corpus:list,每一个元素又是一个list,每个小list中的每一项为：(词汇序号,出现次数)
corpus=[dic.doc2bow(text) for text in train_set]
#print(corpus)
tfidf=models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]
#print(corpus_tfidf)
lda=models.LdaModel(corpus_tfidf,id2word=dic,num_topics=4,alpha=0.05)
corpus_lda=lda[corpus_tfidf]
for i in range(5):
    print("第{}篇文档的话题为：{} ，其top词表示为:{}".format(i,lda.print_topics(i)[0][0],lda.print_topics(i)[0][1]))















