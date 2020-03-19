from pathlib import Path
import pandas as pd
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from sklearn import tree
from joblib import dump, load
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
		acc = np.sqrt(np.sum(acc**2, axis=1)) - 9.8
		x.append([np.mean(acc),np.min(acc), 
				  np.max(acc), np.std(acc), get_energy(acc)])
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
fig, axs = plt.subplots(1, len(FEATS), figsize=(20,5))
for i,feat in enumerate(FEATS):
	axs[i].set_title(feat)
	axs[i].scatter(y_trn, x_trn[:,i])
	axs[i].set_xticks(range(3))
fig.savefig('1_features.png')

if Path('dtree.joblib').exists():
	dtree = load('dtree.joblib')
else:
	dtree = tree.DecisionTreeClassifier()
	dtree = dtree.fit(x_trn, y_trn)
	dot_data = tree.export_graphviz(dtree, 
		class_names=list(class_to_label.keys()),
		feature_names = FEATS,
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

