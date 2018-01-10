# coding=utf-8
import logging
import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os

from collections import OrderedDict
#获取当前路径
path=os.getcwd()
#导入日志配置文件
logging.config.fileConfig('logging.conf')
#创建日志对象
logger=logging.getLogger()

#导入配置文件
conf=ConfigParser.ConfigParser()
conf.read('setting.conf')
#文件路径
trainfile = os.path.join(path,os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","wordidmapfile")))
thetafile = os.path.join(path,os.path.normpath(conf.get("filepath","thetafile")))
phifile = os.path.join(path,os.path.normpath(conf.get("filepath","phifile")))
paramfile = os.path.join(path,os.path.normpath(conf.get("filepath","paramfile")))
topNfile = os.path.join(path,os.path.normpath(conf.get("filepath","topNfile")))
tassginfile = os.path.join(path,os.path.normpath(conf.get("filepath","tassginfile")))

#模型初始参数
K = int(conf.get("model_args","K"))
alpha = float(conf.get("model_args","alpha"))
beta = float(conf.get("model_args","beta"))
iter_times = int(conf.get("model_args","iter_times"))
top_words_num = int(conf.get("model_args","top_words_num"))

class Document(object):
    def __init__(self):
        self.words=[]
        self.length=0

class DataPreProcessing(object):
    def __init__(self):
        self.docs_count=0
        self.words_count=0
        self.docs=[]
        self.word2id=OrderedDict()

    def cachewordidmap(self):
        with codecs.open(wordidmapfile,'w','utf-8') as f:
            for word,id in self.word2id.items():
                f.write(word+'\t'+str(id)+'\n')

class LDAModel(object):
    def __init__(self,dpre):
        self.dpre=dpre#获取预处理参数

        #模型参数
        #聚类个数K，
        # 迭代次数iter_times,
        # 每个类特征词个数top_words_num,
        # 超参数α（alpha） β(beta)
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num

        #文件变量
        #分好词的文件trainfile
        #词对应id文件wordidmapfile
        #文章-主题分布文件thetafile
        #词-主题分布文件phifile
        #每个主题topN词文件topNfile
        #最后分派结果文件tassginfile
        #模型训练选择的参数文件paramfile
        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile

        # p,概率向量 double类型，存储采样的临时变量
        # nw,词word在主题topic上的分布
        # nwsum,每各topic的词的总数
        # nd,每个doc中各个topic的词的总数
        # ndsum,每各doc中词的总数
        self.p = np.zeros(self.K)
        self.nw = np.zeros((self.dpre.words_count,self.K),dtype="int")
        self.nwsum = np.zeros(self.K,dtype="int")
        self.nd = np.zeros((self.dpre.docs_count,self.K),dtype="int")
        self.ndsum = np.zeros(dpre.docs_count,dtype="int")
        self.Z = np.array([ [0 for y in xrange(dpre.docs[x].length)]
                            for x in xrange(dpre.docs_count)])
        # M*doc.size()，文档中词的主题分布,M为文档数

        #随机分配类型
        for x in xrange(len(self.Z)):
            self.ndsum[x]=self.dpre.docs[x].length
            for y in xrange(self.dpre.docs[x].length):
                topic=random.randint(0,self.K-1)
                self.Z[x][y]=topic#为每一个文档中的每一个词随机分配一个topic序号
                self.nw[self.dpre.docs[x].words[y]][topic]+=1
                self.nd[x][topic]+=1
                self.nwsum[topic]+=1
        self.theta=np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count)])
        #print(self.theta.shape)#文档数*主题数，每一行为一个文档，每一的维数为K(topic数)
        self.phi=np.array([[0.0 for y in xrange(self.dpre.words_count)]for x in xrange(self.K)])

    def sampling(self,i,j):
        topic=self.Z[i][j]
        word=self.dpre.docs[i].words[j]
        print self.nw[word]
        #去除掉当前词，所以将牵扯到词的地方都减1
        self.nw[word][topic]-=1
        self.nwsum[topic]-=1
        self.nd[i][topic]-=1
        self.ndsum[i]-=1

        Vbeta=self.dpre.words_count*self.beta
        Kalpha=self.K*self.alpha
        self.p=(self.nw[word]+self.beta)/(self.nwsum+Vbeta)*\
        (self.nd[i]+self.alpha)/(self.ndsum[i]+Kalpha)
        print(self.p)
        #这里不太理解为什么要累加？
        for k in xrange(1,self.K):
            self.p[k]+=self.p[k-1]
        print self.p

        u=random.uniform(0,self.p[self.K-1])
        for topic in xrange(self.K):
            if self.p[topic]>u:
                print('Break!')
                break
        self.nw[word][topic]+=1
        self.nwsum[topic]+=1
        self.nd[i][topic]+=1
        self.ndsum[i]+=1
        return topic

    def est(self):
        self.sampling(1,1)
        # logger.info('开始迭代采样，迭代次数为：%d'%self.iter_times)
        # for x in xrange(self.iter_times):
        #     for i in xrange(self.dpre.docs_count):
        #         for j in xrange(self.dpre.docs[i].length):
        #             topic=self.sampling(i,j)
        #             self.Z[i][j]=topic
        # logger.info('迭代完成')
        # logger.debug('计算文章-主题分布')
        # self._theta()
        # logger.debug('计算词-主题分布')
        # self._phi()
        # logger.debug('保存模型')
        # self.save()





def preprocessing():
    logger.info('载入数据.......')
    with codecs.open(trainfile,'r','utf-8') as f:
        docs=f.readlines()
    logger.info('载入完成，准备生成字典对象和统计文本数据。')
    dpre=DataPreProcessing()
    items_idx=0
    for line in docs:
        if line!=""or line!="\n":
            tmp=line.strip().split()
            doc=Document()
            for item in tmp:
                if dpre.word2id.has_key(item):
                    doc.words.append(dpre.word2id[item])
                else:
                    dpre.word2id[item]=items_idx
                    doc.words.append(dpre.word2id[item])
                    items_idx+=1
            doc.length=len(tmp)
            #print len(tmp)
            dpre.docs.append(doc)
        else:
            pass
    dpre.docs_count=len(dpre.docs)
    dpre.words_count=len(dpre.word2id)
    logger.info('共有%s个文档'%dpre.docs_count)
    dpre.cachewordidmap()#建立词到数字的映射文件
    logger.info('词与序号对应关系已经保存到%s'%wordidmapfile)
    return dpre

def run():
    dpre=preprocessing()
    lda=LDAModel(dpre)
    lda.est()



if __name__ == '__main__':
    run()




























