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
   python HNGDTI.py
   

