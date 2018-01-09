import numpy as np
import lda
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