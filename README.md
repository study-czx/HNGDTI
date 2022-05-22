HNGDTI: A drug-target interaction prediction framework based on heterogeneous network and gene ontology annotations
====
 First, we extracted the representations of drugs and proteins in the heterogeneous network with the heterogeneous graph neural networks. Furthermore, considering the correlation between GO terms, we utilized the weighted graph neural networks in the GO term semantic similarity networks to enhance the representations of GO terms, which were passed to proteins via GO term-protein bipartite networks with graph neural networks. Finally, we concatenated the final drug and protein representations and fed them into the deep neural network.
    
The environment of HNGDTI
===
    python 3.8.8
    cuda 10.2 - cudnn 7.6.5
    pytorch 1.10.0
    dgl 0.6.1
    scikit-learn 0.22.1
Usage
===
    All data are csv files of binary relational data
    Unzip the folders DTI-rand.rar, DTI-net.rar, neg3-9.rar, neg-b3-9.rar and GO.rar
    python HNGDTI.py
    For different settings in the paper, run xxx.py file with a different name (such as Only_add_GO.py).
    
   

