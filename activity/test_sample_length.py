from pathlib import Path
import pandas as pd
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
from sklearn import metrics


def get_energy(X):
    N = X.shape[0]
    fs = 100
    dt = 1/fs
    df = fs/N
    y = np.fft.fft(X)*dt
    energy = np.sum(np.abs(X)**2, axis=0)*df
    return energy

def get_features(ts):
    return [get_energy(ts),ts.min(), ts.max(), ts.std()], ['energy', 'min','max', 'std']

def split_data(ts, chunk_size, activity):
    samples = []
    for i in range(len(ts)):
       if i*chunk_size+chunk_size < len(ts):
        sample = ts[i*chunk_size:i*chunk_size+chunk_size]
        features, column_names = get_features(sample)
        samples.append(features)
    if len(samples)==0:
        return pd.DataFrame()
    df = pd.DataFrame(samples)
    df.columns = column_names
    df['activity'] = activity
    return(df)
    
def get_training_and_test_sets(x, y, split):
    if len(x)!=len(y):
        raise ValueError('x and y are not the same length')
    length = len(x)
    nbr_of_samples_train = int(split[0] * length)
    nbr_samples_test = int(split[1] * length)

    dataset_labels = np.zeros(len(x))
    # set_labels[:set_a_len] = 0 #"traing_set"
    dataset_labels[nbr_of_samples_train:nbr_of_samples_train + nbr_samples_test] = 1  # test

    np.random.shuffle(dataset_labels)

    x_train = x[dataset_labels==0]
    y_train = y[dataset_labels==0]

    x_test = x[dataset_labels==1]
    y_test = y[dataset_labels==1]


    return x_train, y_train, x_test, y_test

class_to_label = {
	'standing': 0,
	'walking' : 1,
	'running' : 2,
}


data_root 	= Path('./data')
files 		= list(data_root.glob('**/*.txt'))

sample_lengths = [10, 50, 100, 250, 500, 750, 1000, 1750, 2500, 5000, 10000]
results = []
for sample_length in sample_lengths:
    print(sample_length)
    df_list = []
    for file in files:
        class_name = str(file).split('/')[-2]
        activity = class_to_label[class_name]
        col_names  = ['%s' % i for i in range(8)]
        df = pd.read_csv(str(file), names=col_names, sep='\t')	
        df_acc = df.loc[df['1'] == 'ACC'].iloc[:,2:5]
        df_acc.columns = ['x','y','z']
        df_acc['xyz'] = ((df_acc['x'].astype(float)**2+df_acc['y'].astype(float)**2+df_acc['z'].astype(float)**2)**.5)
    
        df_list.append(split_data(df_acc['xyz'], sample_length, activity))
    
    df = pd.concat(df_list)
    




    for i in range(100):
        print(i)
        x_train, y_train, x_test, y_test = get_training_and_test_sets(df.drop('activity', axis=1).values,df.activity,[.7,.3])
    
    
        dtree = tree.DecisionTreeClassifier()
        dtree = dtree.fit(x_train, y_train)
        
        
        y_pred_class = dtree.predict(x_test)
        score = metrics.accuracy_score(y_test, y_pred_class)

        results.append([sample_length,score])

df_results=pd.DataFrame(results)
df_results.columns = ['sample_size','score']

df_res_grouped = df_results.groupby('sample_size',as_index=False).agg('mean')

plt.figure()
plt.semilogx(df_res_grouped.sample_size,df_res_grouped.score)
plt.xlabel('sample_length')
plt.ylabel('accuracy')


X = [df_results[df_results['sample_size']==x_i]['score'] for x_i in df_results.sample_size.unique()]

plt.figure()
plt.boxplot(X, labels=df_results.sample_size.unique())
plt.xlabel('sample_length')
plt.ylabel('accuracy')

