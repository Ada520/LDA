# -*- encoding:utf-8 -*-
'''
1. 引言¶

许多数据分析应用都会涉及到从短文本中提取出潜在的主题，比如微博、短信、日志文件或者评论数据。
一方面，提取出潜在的主题有助于下一步的分析，比如情感评分或者文本分类模型。另一方面，短文本数据存在一定的特殊性，我们无法直接用传统的主题模型算法来处理它。短文本数据的主要难点在于：

短文本数据中经常存在多词一义的现象[1]，比如 “dollar”, "$", "$$", "fee", "charges" 拥有相同的含义，但是受限于文本篇幅的原因，我们很难直接从短文本数据中提取出这些信息。
与长文档不同的地方在于，短文本数据中通常只包含一个主题。这看似很好处理，但是传统的主题模型算法都假设一篇文档中包含多个主题，这给建模分析带来了不小的麻烦。
主题提取模型通常包含多个流程，比如文本预处理、文本向量化、主题挖掘和主题表示过程。每个流程中都有多种处理方法，不同的组合方法将会产生不同的建模结果。

本文将主要从实际操作的角度来介绍不同的短文本主题建模算法的优缺点，更多理论上的探讨可以参考以下文章。

下文中我将自己创建一个数据集，并利用 Python scikit-learn 来拟合相应的主题模型。
2\主题模型发现：
本文主要介绍三个主题模型, LDA(Latent Dirichlet Allocation), NMF(Non-Negative Matrix Factorization)和SVD(Singular Value Decomposition)。
本文主要采用 scikit-learn 来实现这三个模型。

除了这三个模型外，还有其他一些模型也可以用来发现文档的结构。
其中最重要的一个模型就是 KMeans 聚类模型，本文将对比 KMeans 聚类模型和其他主题模型的拟合效果。
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,NMF,LatentDirichletAllocation
from sklearn.cluster import KMeans
#首先我们需要构建文本数据集。本文将以四个自己构建的文本数据集为例来构建主题模型：
'''
clearcut topics: 该数据集中只包含两个主题—— "berger-lovers" 和 "sandwich-haters"。
unbalanced topics: 该数据集与第一个数据集包含的主题信息一致，但是此数据集的分布是有偏的。
semantic topics: 该数据集包含四个主题，分别是 "berger-lovers"， "berger-haters"，"sandwich-lovers" 和 "sandwich-haters"。
此外，该数据集中还包含了两个潜在的主题 “food” 和 “feelings”。
noisy topics: 正如前文所说的，短文本数据中经常存在多词一义的现象，该数据集主要用于模拟两个主题不同类型的文本。
该数据集文本的篇幅小于其他三个数据集，这可以用来检验模型是否能够很好地处理短文本数据。
'''
def generate_clearcut_topics():
    #生成1000个‘we love bergers'1000个'we hate sandwiches'
    return np.repeat(['we love bergers','we hate sandwiches'],[1000,1000])


def generate_unbalanced_topics():
    #生成10个‘we love bergers'1000个'we hate sandwiches'
    return np.repeat(['we love bergers','we hate sandwiches'],[10,1000])

def generate_semantic_context_topics():
    #生成1000个‘we love bergers'1000个'we hate sandwiches' 1000个'we hate bergers'1000个'we love bergers'
    return np.repeat(['we love bergers','we hate bergers','we love snadwiches','we hate sandwiches'],1000)
def generate_noisy_topics():
    def _random_typos(word,n):
        type_index=np.random.randint(0,len(word),n)
        return [word[:i]+"X"+word[i+1:] for i in type_index]
    t1=['we love %s'% w for w in _random_typos('bergers',15)]
    t2=['we hate %s' % w for w in _random_typos('sandwiches',15)]
    return np.r_[t1,t2]

sample_text={
    "clearcut topics": generate_clearcut_topics()
    , "unbalanced topics": generate_unbalanced_topics()
    , "semantic topics": generate_semantic_context_topics()
    , "noisy topics": generate_noisy_topics()
}

from collections import Counter

def find_topic(texts,topic_model,n_topics,vec_model='tf',thr=0.02,**kwargs):
    """Return a list of topics from texts by topic models - for demostration of simple data
    texts: array-like strings
    topic_model: {"nmf", "svd", "lda", "kmeans"} for LSA_NMF, LSA_SVD, LDA, KMEANS (not actually a topic model)
    n_topics: # of topics in texts
    vec_model: {"tf", "tfidf"} for term_freq, term_freq_inverse_doc_freq
    thr: threshold for finding keywords in a topic model
    """

    #1. vectorization
    vectorizer=CountVectorizer() if vec_model=="tf" else TfidfVectorizer
    text_vec=vectorizer.fit_transform(texts)
    #print text_vec#这是一个矩阵，每一行代表一个文本，每一列代表一个词对应的序号，值为出现的次数
    words=np.array(vectorizer.get_feature_names())
    #print words
    #2.topic finding
    topic_models={"nmf":NMF,"svd":TruncatedSVD,"lda":LatentDirichletAllocation,"kmeans":KMeans}
    topicfinder=topic_models[topic_model](n_topics,**kwargs).fit(text_vec)
    topic_dists=topicfinder.components_ if topic_model is not 'kmeans' else topicfinder.cluster_centers_
    topic_dists/=topic_dists.max(axis=1).reshape((-1,1))
    ## 3. keywords for topics
    ## Unlike other models, LSA_SVD will generate both positive and negative values in topic_word distribution,
    ## which makes it more ambiguous to choose keywords for topics. The sign of the weights are kept with the
    ## words for a demostration here
    def _topic_keywords(topic_dist):
        keywords_index=np.abs(topic_dist)>=thr
        keywords_prefix=np.where(np.sign(topic_dist)>0,"","^")[keywords_index]
        keywords="|".join(map(lambda x:"".join(x),zip(keywords_prefix,words[keywords_index])))
        return keywords
    topic_keywords=map(_topic_keywords,topic_dists)
    return "\n".join("Topic % i: %s"%(i,t) for i ,t in enumerate(topic_keywords))
print(find_topic(sample_text["clearcut topics"], "lda", 4, vec_model="tf"))























