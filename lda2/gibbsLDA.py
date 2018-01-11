# -*- coding:utf-8 -*-

import random
import argparse
import collections
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs

class GibbsLDA:
    def __init__(self,K,alpha,beta):
        self.alpha=alpha
        self.beta=beta
        self.K=K
        self._init_list=lambda x,y:[y for i in range(x)]
        self._init_array=lambda x,y,z:[[z for j in range(y)]for i in range(x)]

    def set_vocabulary(self,vacabulary):
        self.vocabulary=vacabulary
        self.V=len(self.vocabulary)
        self.word_list=self._init_list(self.V,None)
        for word,word_id in self.vocabulary.items():
            self.word_list[word_id]=word
        self.topic_word=self._init_array(self.K,self.V,0)#K*V
        self.words_of_topic=self._init_list(self.K,0)

    def one_iteration(self):
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            for word_id in range(len(doc)):
                word=doc[word_id]
                topic=self.assignments[doc_id][word_id]
                self.doc_topic[doc_id][topic]-=1
                self.topic_word[topic][word]-=1
                self.words_of_topic[topic]-=1
                #print len(self.doc_topic[doc_id])
                ps=[]
                for topic in range(self.K):
                    ps.append((self.doc_topic[doc_id][topic]+self.alpha)*
                    (self.topic_word[topic][word]+self.beta)/
                    (self.words_of_topic[topic]+len(self.vocabulary)*self.beta))
                print ps
                x=sum(ps)*random.random()
                topic=0
                acc=0
                for p in ps:
                    acc+=p


    def loop(self,docs,burnin,iteration):
        self.docs=docs
        self.M=len(self.docs)
        self.assignments=[[0 for i in range(len(doc))] for doc in self.docs]#将docs中所有的值置为0
        self.doc_topic=self._init_array(self.M,self.K,0)#M*K
        for doc_id in range(len(self.docs)):
            doc=self.docs[doc_id]
            for word_id in range(len(doc)):
                word=doc[word_id]
                topic=random.randrange(0,self.K)
                self.doc_topic[doc_id][topic]+=1#文档-主题
                self.topic_word[topic][word]+=1#主题-词
                self.words_of_topic[topic]+=1
                self.assignments[doc_id][word_id]=topic
        #init phi and theta
        self.phi=self._init_array(self.K,self.V,0)#主题-词
        self.theta=self._init_array(self.M,self.K,0)#文章-主题

        # sampling loop
        for it in range(burnin+iteration):
            print('第%s轮迭代开始...'%(it+1))
            self.one_iteration()



    def load(self,modelfile):
        ofline=open(modelfile)
        self.alpha,self.beta=ofline.readlines().split()
        print self


def load(docfile):
    #n_stopwords:设定为stopwords的词频，即当词频大于n_stopwords的时候，则认为改词为一个停用词
    #n_words:想要保留的词的词频阈值，在此设置为4，可以根据语料的大小进行设置
    docs=[]
    with codecs.open(docfile,'r','utf-8') as f:
        for line in f:
            docs.append(line.split())
    ## filter stopwords and tail words
    counter=collections.Counter()
    for doc in docs:
        counter.update(doc)
    #print counter
    words=[w for w,number in counter.items() if number>=2 and number<=8 ]
    words=set(words)

    vovabulary={}
    for i,doc in enumerate(docs):
        for word in doc:
            if word not in words:
                continue
            if word not in vovabulary:
                vovabulary[word]=len(vovabulary)
        docs[i]=[vovabulary[word] for word in doc if word in vovabulary]
    return docs,vovabulary

def load_with_v(docfile,vocabulary):
    docs=[line.split(',')for line in open(docfile)]
    for i,doc in enumerate(docs):
        docs[i]=[vocabulary[word] for word in doc if word in vocabulary]
    return docs








if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='LDA的参数')
    # parser.add_argument('--train',type=str, help=u'用于训练的文本集，每行代表一个文档，文档中的词用空格隔开')
    # parser.add_argument('--predict',type=str, help='')
    # parser.add_argument('--model',type=str, help='')
    # parser.add_argument('--result',type=str, help='')
    # parser.add_argument('--burnin',type=int,default=30, help='')
    # parser.add_argument('--iteration',type=int,default=5, help='')
    # parser.add_argument('--n_stops',type=int,default=100, help=u'设定停用词个数')
    # parser.add_argument('--n_words',type=int,default=1000, help=u'设定使用的词的个数')
    # parser.add_argument('-K',type=int,default=20, help=u'主题个数')
    # parser.add_argument('--alpha',type=int,default=1, help='')
    # parser.add_argument('--beta',type=int,default=1, help='')
    # args = parser.parse_args()
    # # if args.train:
    # #     docs,vocabulary=load(args.train,args.n_stop,args.n_words)
    # #     model=GibbsLDA(args.K,args)
    trainfile='ying.txt'
    docs,vocabulary=load(trainfile)
    #docs是一个列表，每一项表示为一个文档，文档中表示对应的词序号

    model=GibbsLDA(5,0.02,0.02)
    model.set_vocabulary(vocabulary)
    model.loop(docs,3,3)









































