import numpy as np
import os
import argparse
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics.pairwise import euclidean_distances
import torch
from tqdm import tqdm

def Norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def create_dataframe_from_pt_npy(pt_path, npy_path):
    """
    Process a single .pt and .npy file pair, generating a DataFrame.
    """
    print(f"Loading: {os.path.basename(pt_path)}")
    pt_data = torch.load(pt_path, map_location='cpu')
    features = pt_data['features'].numpy()
    coords = pt_data['coords']

    print(f"Loading: {os.path.basename(npy_path)}")
    prob_map = np.load(npy_path)
    rows, cols, _ = prob_map.shape
    processed_data = []


    class_mapping = {
       
        0: 0,  # 'Low-Grade Urothelial Carcinoma' 
        1: 1,  # 'High-Grade Urothelial Carcinoma' 
        2: 2,  # 'Urothelial carcinoma with histologic variants'
        4: 3,  # 'Lamina propria invasion'
        5: 4,  # 'Muscularis propria invasion'
        3: 5,  # 'Lymphovascular invasion' 
        6: 6,  # 'Renal parenchyma invasion'
        7: 7,  # 'Adipose tissue invasion' 
    }

    for i in tqdm(range(len(coords))):
        x, y = coords[i]
        if y < prob_map.shape[0] and x < prob_map.shape[1]:
            if all(prob_map[y, x, :] == 0):
                continue
            prob_vector = prob_map[y, x, :]
            original_class = np.argmax(prob_vector)

            pred_class = class_mapping.get(original_class, original_class)
            feature_vector = features[i]
            row_data = [x, y, pred_class]
            row_data.extend(feature_vector)
            processed_data.append(row_data)
        else:
            print(f"Warning: Coordinates ({x}, {y}) are outside the probability map range {prob_map.shape[:2]}, skipped.")

    num_features = features.shape[1]
    col_names = ['n_col', 'n_row', 'pred_class'] + [f'feature_{j}' for j in range(num_features)]
    df = pd.DataFrame(processed_data, columns=col_names)

    return df, rows, cols

def paga(wsi_df):
 
    x, obs = pd.DataFrame(wsi_df.iloc[:,3:]).astype('float64'), pd.DataFrame(wsi_df.iloc[:,0:3])
    adata = ad.AnnData(x,obs)
    
    print('start paga')
    sc.tl.pca(adata, svd_solver='arpack') ## dimension reduction
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.leiden(adata,resolution=1) ## clustering
    sc.tl.paga(adata, groups='leiden') ## PAGA for clustering partition

    #start_class = adata.obs['pred_class'].unique().min()
    start_class = 0
    start_indices = np.flatnonzero(adata.obs['pred_class'] == start_class)
    
    if len(start_indices) > 0:

        start_node_pca = adata.obsm['X_pca'][start_indices].mean(axis=0)
        

        from sklearn.metrics.pairwise import euclidean_distances
        distances_to_centroid = euclidean_distances(adata.obsm['X_pca'], start_node_pca.reshape(1, -1))
        root_node_index = np.argmin(distances_to_centroid)
        adata.uns['iroot'] = root_node_index
    else:
        # If there is no image block of category 0, select the node with the highest degree in the pseudo-time graph as an alternative
        print("Warning: No starting point of category 0 found, using default strategy.")
        adata.uns['iroot'] = np.flatnonzero(adata.obs['pred_class'] == adata.obs['pred_class'].unique().min())[0]

    
    sc.tl.dpt(adata)  ### DPT to calculate psudotime
    print('end paga')
    
    ## Calculate euclidean distance
    print('start Euclidean')
    coordiantes = adata.obs.sort_values(by="dpt_pseudotime")[["n_col","n_row"]]
    dist =  pd.DataFrame(euclidean_distances(coordiantes,coordiantes))
    dist = Norm(dist[0])
    dist.index, dist.name = coordiantes.index, "Distance"
    adata.obs = adata.obs.join(dist)
    print('end Euclidean')
    
    return adata

if __name__ == "__main__":
    pt_path = "path/to/your/file.pt"
    npy_path = "path/to/your/file.npy"
    output_path= "path/to/output"

    pt_files = [f for f in os.listdir(pt_path) if f.endswith('.pt')]

    for pt_file in pt_files:
        base_name = os.path.splitext(pt_file)[0].replace('_features', '')
        pt_full_path = os.path.join(pt_path, pt_file)
        npy_full_path = os.path.join(npy_path, f"{base_name}.npy")

        if not os.path.exists(npy_full_path):
            print(f"Error: No corresponding .npy file found: {npy_full_path}, skipping.")
            continue
        
        try:
 
            wsi_df, rows, cols = create_dataframe_from_pt_npy(pt_full_path, npy_full_path)
            
            if wsi_df.empty:
                print("Error: The generated DataFrame is empty, unable to proceed.")
                continue

            # 2. Perform PAGA and pseudotime analysis
            adata = paga(wsi_df)
            
            output_filename = f"{base_name}_{cols}cols_{rows}rows.h5ad"
            output_h5ad_path = os.path.join(output_path, output_filename)
            adata.write(output_h5ad_path, compression="gzip")
            print(f"Successfully saved results to: {output_h5ad_path}")

            print("Creating and saving pseudotime heatmap...")
            # 1. Create an empty canvas, using np.nan as the background value
            # Note: prob_map has the shape (rows, cols, _), so the heatmap is (rows, cols)
            pseudotime_heatmap = np.full((rows, cols), np.nan)

            # 2. Iterate over adata.obs and fill the canvas with pseudotime values
            for index, row in adata.obs.iterrows():
                r, c = int(row['n_row']), int(row['n_col'])
                pseudotime_value = row['dpt_pseudotime']
                if r < rows and c < cols:
                    pseudotime_heatmap[r, c] = pseudotime_value

            # 3. Define output path and save
            output_npy_filename = f"{base_name}_pseudotime_map.npy"
            output_npy_path = os.path.join(output_path, output_npy_filename)
            np.save(output_npy_path, pseudotime_heatmap)
            print(f"Successfully saved pseudotime heatmap to: {output_npy_path}")
            # --- 新增代码结束 ---

        except Exception as e:
            print(f"Error occurred while processing file {pt_file}: {e}")

