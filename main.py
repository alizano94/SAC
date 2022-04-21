import os
from src.control import RL

full_features_csv = "full_raw_features.csv"
projected_features_csv = "full_umap_2D_features.csv"
clusters_csv = 'full_'

control = RL(w=100,m=1,a=4)
control.raw_featInception(out_name=full_features_csv,
                            data_set='full')
control.umap(n=2,load_file=full_features_csv,
            out_file=projected_features_csv)
control.cluster_hdbscan(mcs=40,ms=5,eps=0.1,
                        load_file=projected_features_csv,
                        out_file=clusters_csv)
