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
files 		= data_root.glob('**/*.txt')

one_phone   = False

acc_feats = []
activity  = []
for file in files:
	if one_phone and (not ('ddd' in str(file))):
		continue
	class_name = str(file).split('/')[-2]
	activity.append(class_to_label[class_name])
	col_names  = ['%s' % i for i in range(8)]
	df = pd.read_csv(str(file), names=col_names, sep='\t')	
	acc = df.loc[df['1'] == 'ACC'].iloc[:,2:5].to_numpy().astype(np.double)	
	acc_feats.append(np.concatenate([np.mean(acc,axis=0),
						np.min(acc,axis=0),
						np.max(acc,axis=0),
						np.std(acc,axis=0),						
						get_energy(acc)]))
	# if not ('uncal' in str(file)):
	# 	gyr = df.loc[df['1'] == 'GYR'].iloc[:,2:5].to_numpy().astype(np.double)
	# 	gyr_erg = get_energy(gyr)

	# if ('rf' in str(file)):
	# 	rss_cell = df.loc[df['1'] == 'RSSCELL'].iloc[:,7].to_numpy().astype(np.double)
	# 	rss_cell_std = np.std(rss_cell)
	# 	rss_wifi = df.loc[df['1'] == 'RSSWIFI'].iloc[:,5].to_numpy().astype(np.double)
	# 	rss_wifi_std = np.std(rss_wifi)
	# 	# print(rss_cell_std, rss_wifi_std, class_to_label[class_name])
acc_feats = np.vstack(acc_feats)
fig, axs = plt.subplots(len(FEATS), 3, figsize=(20,10))
for i,feat in enumerate(FEATS):
	axs[i,0].set_ylabel(feat)
	for j in range(3):
		ax = axs[i,j]
		ax.scatter(activity, acc_feats[:,(i*3+j)])
		ax.set_xticks(range(3))
fig.savefig('1_features.png')

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(acc_feats, activity)
dot_data = tree.export_graphviz(dtree, 
	class_names=list(class_to_label.keys()),
	feature_names = DFEATS,
	out_file=None) 
graph = graphviz.Source(dot_data)
if one_phone:
	graph.render('activity_ddd') 
else:
	graph.render('activity') 