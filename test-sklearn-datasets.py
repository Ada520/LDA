import sklearn.datasets

'''
Scikit已经包含了定制的读取器来读取这个数据集。在读取数据时，
可以设置环境变量MLCOMP_DATASETS_HOME，
或者通过mlcomp_root参数直接指定路径：
'''
MLCOMP_DIR=r'./datasets'
dataset=sklearn.datasets.load_mlcomp('20news-18828','train',mlcomp_root=MLCOMP_DIR)
# print dataset.filenames
# print 'Number of posts:',len(dataset.filenames)
 # 通过设置函数load_mlcomp的第2个参数”train”、”predict”可以选取训练集或测试集。
 #    为方便起见，把范围限制在某些新闻组中，使整个实验流程更短。
# 我们可以通过设置categories参数实现这一点：

groups = ['comp.graphics', 'comp.os.ms-windows.misc']
dataset=sklearn.datasets.load_mlcomp('20news-18828','train',mlcomp_root=MLCOMP_DIR,
                                     categories=groups)
print (len(dataset.data))
print (dataset.data[1])