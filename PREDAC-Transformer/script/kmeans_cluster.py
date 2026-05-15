import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import os,sys
import matplotlib.pyplot as plt
import umap
from sys import argv
import time
import argparse
import pandas as pd
import numpy as np
from functools import reduce
import itertools
def read_csv_feature(filePath):
    # 读取文件
    f = open(filePath, encoding='utf-8')
    reader = pd.read_csv(f, sep=',', iterator=True)#\t
    loop = True
    chunkSize = 10000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iteration is END!!!')
    df = pd.concat(chunks, axis=0, ignore_index=True)
    f.close()
    return df 

parser = argparse.ArgumentParser()

parser.add_argument('--inputcsv', required=True, metavar='file')
parser.add_argument('--subtype', required=True, metavar='subtype')
parser.add_argument('--outdir', required=True, metavar='file')
parser.add_argument('--load_data', default='False',required=False, metavar='file')
args = parser.parse_args()

start_time=time.time()

# 设置自定义保存路径
save_path = args.outdir
os.makedirs(save_path, exist_ok=True)  # 如果路径不存在，则创建路径
subtype = args.subtype

df=read_csv_feature(args.inputcsv)
print(df)
df['strain_time_diff']=abs(df['year_2']-df['year_1'])
df['prob'][df['strain_time_diff']>15]=1
df=df.reset_index(drop=True)
print(len(df))
strains = pd.concat([df['new_name_1'], df['new_name_2']]).unique()
strain_to_index = {strain: index for index, strain in enumerate(strains)}

##generate embedding by umap
if args.load_data!='False':
    umap_embedding=np.load(args.load_data)
    print(umap_embedding.shape)
else:
    n_strains = len(strains)
    similarity_matrix = np.zeros((n_strains, n_strains))
    # distance matrix
    similarity_matrix = 1 - similarity_matrix

    for index, row in df.iterrows():
        i = strain_to_index[row['new_name_1']]
        j = strain_to_index[row['new_name_2']]
        similarity = row['prob']
        if i < n_strains and j < n_strains:  
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    if subtype == 'H3N2':  
        reducer = umap.UMAP(n_components=3, n_neighbors=18, min_dist=0.004, init="spectral", random_state=42)#, low_memory=True
    if subtype == 'H1N1':  
        reducer = umap.UMAP(n_components=3, n_neighbors=40, min_dist=0.3, init="spectral", random_state=42)
    umap_embedding = reducer.fit_transform(similarity_matrix)
    np.save(os.path.join(save_path,'embeddings_umap.npy'), umap_embedding)

#kmeans for cluster
scaled_data=umap_embedding
results = []
all_cluster_results = []
k_values = range(2, 41)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42 )
    kmeans.fit(scaled_data)
    
    labels = kmeans.labels_
   
    silhouette_avg = silhouette_score(scaled_data, labels)
    ch_score = calinski_harabasz_score(scaled_data, labels)
    wcss = kmeans.inertia_

    results.append({
        'k': k,
        'Silhouette Score': silhouette_avg,
        'CH Index': ch_score,
        'WCSS': wcss
    })
    
    cluster_result_df = pd.DataFrame({
        'name': strains,
        'k_'+str(k): labels   
    } )
    cluster_result_df.set_index('name', inplace=True)
    cluster_result_df.to_csv(os.path.join(save_path, f'cluster_results_k{k}.csv'))
    all_cluster_results.append(cluster_result_df)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(os.path.join(save_path, 'kmeans_evaluation.csv'))


all_cluster_results_df = pd.concat(all_cluster_results,axis=1)
all_cluster_results_df.to_csv(os.path.join(save_path, 'kmeans_clustering_results.csv'))

wcss_values = [result['WCSS'] for result in results]
plt.figure(figsize=(8, 6))
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(False)
plt.savefig(os.path.join(save_path, 'wcss_plot.png'))
plt.close()


best_k_silhouette = results_df.loc[results_df['Silhouette Score'].idxmax(), 'k']
best_k_ch = results_df.loc[results_df['CH Index'].idxmax(), 'k']
best_k_wcss = results_df.loc[results_df['WCSS'].idxmin(), 'k']

with open(os.path.join(save_path, 'kmeans_clustering_results.log'), 'w') as file:
    file.write(f"Best k based on Silhouette Score: {best_k_silhouette}\n")
    file.write(f"Best k based on CH Index: {best_k_ch}\n")
    file.write(f"Best k based on WCSS: {best_k_wcss}\n")
    file.write("alltime is %s"%(time.time()-start_time))

