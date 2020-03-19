from pathlib import Path
import pandas as pd
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 

def get_energy(X):
	N = X.shape[0]
	fs = 100
	dt = 1/fs
	df = fs/N
	y = fftpack.fft(X)*dt
	energy = np.sum(np.abs(X)**2, axis=0)*df
	return energy

def dir2dataset(root):
	x, y = [], []
	for file in root.glob('**/*.txt'):
		class_name = str(file).split('/')[-2]
		y.append(class_to_label[class_name])
		col_names  = ['%s' % i for i in range(8)]
		df = pd.read_csv(str(file), names=col_names, sep='\t')	
		acc = df.loc[df['1'] == 'ACC'].iloc[:,2:5].to_numpy().astype(np.double)	
		x.append(np.concatenate([np.mean(acc,axis=0),
							np.min(acc,axis=0),
							np.max(acc,axis=0),
							np.std(acc,axis=0),						
							get_energy(acc)]))
	x = np.vstack(x)
	return x, y



class_to_label = {
	'standing': 0,
	'walking' : 1,
	'running' : 2,
}

FEATS = ['MEAN', 'MIN', 'MAX', 'STD', 'ERG']

DFEATS = ['MEANX','MEANY', 'MEANZ',
		 'MINX',  'MINY',   'MINZ',
		 'MAXX',  'MAXY',   'MAXZ',
		 'STDX',  'STDY',   'STDZ',
		 'ERGX',  'ERGY',   'ERGZ',]

data_root 	= Path('./data')
train_root  = data_root/'train'
one_phone   = False

x_trn, y_trn = dir2dataset(train_root)
fig, axs = plt.subplots(len(FEATS), 3, figsize=(20,10))
for i,feat in enumerate(FEATS):
	axs[i,0].set_ylabel(feat)
	for j in range(3):
		ax = axs[i,j]
		ax.scatter(y_trn, x_trn[:,(i*3+j)])
		ax.set_xticks(range(3))
fig.savefig('1_features.png')

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(x_trn, y_trn)
dot_data = tree.export_graphviz(dtree, 
	class_names=list(class_to_label.keys()),
	feature_names = DFEATS,
	out_file=None) 
graph = graphviz.Source(dot_data)
if one_phone:
	graph.render('activity_ddd') 
else:
	graph.render('activity')


test_root = data_root/'test'
x_tst, y_tst = dir2dataset(test_root)
yhat_tst = dtree.predict(x_tst)
accuracy = np.count_nonzero(yhat_tst == y_tst)/len(y_tst)
print('Prediction accuracy - %0.3f' % accuracy)

