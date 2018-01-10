# this is a simple test about python_lda
import numpy as np
import lda

'''
X=lda.datasets.load_reuters()
vocab=lda.datasets.load_reuters_vocab()
titles=lda.datasets.load_reuters_titles()

model=lda.LDA(n_topics=20,n_iter=1500,random_state=1)
model.fit(X)#model.fit_transform(X) is also available
topic_word=model.topic_word_# model.componenets_ also works
n_topic_words=8


for i,topic_dist in enumerate(topic_word):
    topic_words=np.array(vocab)[np.argsort(topic_dist)][:-n_topic_words:-1]
    print('Topic {}:{}'.format(i,' '.join(topic_words)))
'''

'''
## The document-topic distribution are available in model.doc_topic_
doc_topic=model.doc_topic_
for i in range(10):
    print(doc_topic[i])
    print ('{}(top topic:{})'.format(titles[i],doc_topic[i].argmax()))
'''
## Document-topic distribution may be inferred for out-of-sample texts using the transform method:
X=lda.datasets.load_reuters()
titles=lda.datasets.load_reuters_titles()
X_train=X[10:]
X_test=X[:10]
titles_test=titles[:10]
model=lda.LDA(n_topics=20,n_iter=1500,random_state=1)
model.fit(X_train)
doc_topic_test=model.transform(X_test)
print (doc_topic_test)
for titles,topics in zip(titles_test,doc_topic_test):
    print('{}(top topic:{})'.format(titles,topics.argmax()))

import matplotlib.pyplot as plt
plt.plot(model.loglikelihoods_[5:])
plt.show()