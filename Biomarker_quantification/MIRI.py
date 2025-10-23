
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances
import anndata as ad
import math
import os
import pandas as pd
import numpy as np
def Norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_pairwise_speed(dist, time_f):
    ##input is distance matrix and pseudotime
    speed_df = np.zeros((len(time_f), len(time_f)))
    speed_df[:] = np.nan
    
    for i in dist.index.values:
        #print(i)
        for j in dist.index.values:
            #print(i,j)
            if i != j :
                v = dist.iloc[i,j]/abs(time_f.loc[i,'dpt_pesudotime'] - time_f.loc[j,'dpt_pesudotime'])
            else:
                v = 0
                
            speed_df[i,j] = v
        
    return speed_df

def calculate_fitness(h5ad_file_path):

    f = os.path.basename(h5ad_file_path)

    try:
        parts = f.split('_')
        n_cols = int(parts[-2][:-4])  
        n_rows = int(parts[-1].split('.')[0][:-4]) 
    except (IndexError, ValueError) as e:
        print(f"error: {e}")

        return {'speed': 0, 'shannon_index': 0}

    adata = ad.read_h5ad(h5ad_file_path)

    col_names = ['norm_col', 'norm_row', 'dpt_pesudotime']
    df = pd.DataFrame(columns=col_names)
    df['norm_col'] = adata.obs.groupby(['leiden','pred_class'])['n_col'].mean()/n_cols
    df['norm_row'] = adata.obs.groupby(['leiden','pred_class'])['n_row'].mean()/n_rows
    df['dpt_pesudotime'] = adata.obs.groupby(['leiden','pred_class'])['dpt_pseudotime'].mean()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=col_names,how='any').reset_index()
    speed = 0

    if not df.empty and df['dpt_pesudotime'].nunique() > 1:
        coordinates = df[["norm_col","norm_row"]]
        dist =  pd.DataFrame(euclidean_distances(coordinates,coordinates))
        dist = Norm(dist)
        speed_df= get_pairwise_speed(dist, df)

        if not pd.DataFrame(speed_df).isnull().all().all():
            speed = pd.DataFrame(speed_df).fillna(0).sum().sum()/len(df)
            speed = math.log(speed) if speed > 0 else 0
    
    return {'speed': speed}


def TIM_Score(prob_map_path, cell_size):
    # prob_map: MxNx8 numpy array contains the probabilities
    # cell_size: number of patch to be consider as one grid-cell
    prob_map = np.load(prob_map_path)
    pred_map = np.zeros((prob_map.shape[0],prob_map.shape[1]))
    for i in range(prob_map.shape[0]):
        for j in range(prob_map.shape[1]):
            pred_map[i][j] = np.argmax(prob_map[i,j,:])
            # print(multi_classes[i][j])
            if prob_map[i,j,0] == 0 and pred_map[i][j] == 0:
                pred_map[i][j] = 1
    T = np.int8(pred_map == 0)  # patches predicted as tumour
    M = np.int8(pred_map == 4)  # patches predicted as musele
    [rows, cols] = T.shape
    stride = np.int32(cell_size / 2)
    t = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    m = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    k = 0
 
    for i in range(0, rows - cell_size + 1, stride):
        for j in range(0, cols - cell_size + 1, stride):
            t[k] = np.mean(np.mean(T[i:i + cell_size, j:j + cell_size]))
            m[k] = np.mean(np.mean(M[i:i + cell_size, j:j + cell_size]))
            k += 1

    index = np.logical_and(t == 0, m == 0)
    index = np.where(index)[0]
    t = np.delete(t, index)
    m = np.delete(m, index)
    tim_score = 0.0
    coloc_score = 0.0 
    if len(t) == 0:  
        tim_score = 0  
    else:
        
        t = t/(t + m)
        m = m/(t + m)
   
        coloc_score = (2 * sum(t*m)) / (sum(t**2) + sum(m**2))
        if np.sum(t) == 0:
            tim_score = 1 
        else:
            l2t_ratio = np.sum(m) / np.sum(t)  
            tim_score = 0.5 * coloc_score * l2t_ratio  
    return tim_score,coloc_score


prob_map_path = "path/to/your/prob_map.npy"
h5ad_map_path = "path/to/your/h5ad_map.h5ad"
cell_size = 10  
TIR_score, Coloc_M = TIM_Score(prob_map_path, cell_size)
speed = calculate_fitness(h5ad_map_path)['speed']
miri_score = speed * Coloc_M



