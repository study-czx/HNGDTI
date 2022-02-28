import numpy as np
import random
import torch
from torch.utils import data
import dgl

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)

def id_map(my_id):
    id_map = {"interger_id": "origin_id"}
    for i in range(len(my_id)):
        id_map[my_id[i]] = i
    return id_map

def Get_sample(DTI, N_DTI, dr_id_map, p_id_map):
    P_list, N_list = [],[]
    P_label, N_label = [],[]
    for i in range(len(DTI)):
        P_list.append([dr_id_map[DTI[i][0]], p_id_map[DTI[i][1]]])
        P_label.append([1])
    for j in range(len(N_DTI)):
        N_list.append([dr_id_map[N_DTI[j][0]], p_id_map[N_DTI[j][1]]])
        N_label.append([0])
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

def Get_Train_sample(DTI, N_DTI, dr_id_map, p_id_map):
    P_list, N_list = [],[]
    P_label, N_label = [],[]
    for i in range(len(DTI)):
        P_list.append([dr_id_map[DTI[i][0]], p_id_map[DTI[i][1]]])
        P_label.append([1])
    for j in range(len(N_DTI)):
        N_list.append([N_DTI[j][0], N_DTI[j][1]])
        N_label.append([0])
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

def Get_index(data, id_map1, id_map2):
    my_list = []
    for i in range(len(data)):
        my_list.append([id_map1[data[i][0]], id_map2[data[i][1]]])
    return my_list

def get_train_loader(X, Y, b_size):
    class Dataset(data.Dataset):
        def __init__(self):
            self.Data = X
            self.Label = Y

        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])
            label = torch.tensor(self.Label[index])
            return txt, label

        def __len__(self):
            return len(self.Data)

    Data = Dataset()
    loader = data.DataLoader(Data, batch_size=b_size, shuffle=True, drop_last=True, num_workers=0)
    return loader

def get_test_loader(X, Y, b_size):
    class Dataset(data.Dataset):
        def __init__(self):
            self.Data = X
            self.Label = Y

        def __getitem__(self, index):
            txt = torch.from_numpy(self.Data[index])
            label = torch.tensor(self.Label[index])
            return txt, label

        def __len__(self):
            return len(self.Data)

    Data = Dataset()
    loader = data.DataLoader(Data, batch_size=b_size, shuffle=False, num_workers=0)
    return loader

def computer_label(input, threshold):
    label = []
    for i in range(len(input)):
        if (input[i] >= threshold):
            y = 1
        else:
            y = 0
        label.append(y)
    return label

def shuffer(X, Y ,seed):
    index = [i for i in range(len(X))]
    np.random.seed(seed)
    np.random.shuffle(index)
    new_X, new_Y = X[index], Y[index]
    return new_X, new_Y

def delete_smalle_sim(sim, remain_ratio):
    data = []
    for i in range(1, len(sim)):
        for j in range(0, i):
            data.append(sim[i][j])
    data.sort(reverse=True)
    number_remain = int(len(data)*remain_ratio)
    number_th = data[number_remain-1]
    sim[sim < number_th] = 0
    for i in range(len(sim)):
        sim[i][i] = 0
    return sim

def Get_GO_Protein_graph(P_MF_data, P_BP_data, P_CC_data, num_nodes_dict_p_mf, num_nodes_dict_p_bp, num_nodes_dict_p_cc):
    MF_P, MF_GO = [], []
    BP_P, BP_GO = [], []
    CC_P, CC_GO = [], []
    for i in range(len(P_MF_data)):
        MF_P.append(P_MF_data[i][0])
        MF_GO.append(P_MF_data[i][1])
    for i in range(len(P_BP_data)):
        BP_P.append(P_BP_data[i][0])
        BP_GO.append(P_BP_data[i][1])
    for i in range(len(P_CC_data)):
        CC_P.append(P_CC_data[i][0])
        CC_GO.append(P_CC_data[i][1])
    MF_src, MF_dst = torch.tensor(MF_P), torch.tensor(MF_GO)
    BP_src, BP_dst = torch.tensor(BP_P), torch.tensor(BP_GO)
    CC_src, CC_dst = torch.tensor(CC_P), torch.tensor(CC_GO)
    # 将关联输入到图中，构建异构图，其中无向边用两条有向边表示
    MF_graph_data = {('protein', 'P-MF', 'MF'): (MF_src, MF_dst)}
    BP_graph_data = {('protein', 'P-BP', 'BP'): (BP_src, BP_dst)}
    CC_graph_data = {('protein', 'P-CC', 'CC'): (CC_src, CC_dst)}
    MF_graph_data2 = {('MF', 'MF-P', 'protein'): (MF_dst, MF_src)}
    BP_graph_data2 = {('BP', 'BP-P', 'protein'): (BP_dst, BP_src)}
    CC_graph_data2 = {('CC', 'CC-P', 'protein'): (CC_dst, CC_src)}
    P_MF_graph = dgl.heterograph(MF_graph_data, num_nodes_dict=num_nodes_dict_p_mf)
    P_BP_graph = dgl.heterograph(BP_graph_data, num_nodes_dict=num_nodes_dict_p_bp)
    P_CC_graph = dgl.heterograph(CC_graph_data, num_nodes_dict=num_nodes_dict_p_cc)
    MF_P_graph = dgl.heterograph(MF_graph_data2, num_nodes_dict=num_nodes_dict_p_mf)
    BP_P_graph = dgl.heterograph(BP_graph_data2, num_nodes_dict=num_nodes_dict_p_bp)
    CC_P_graph = dgl.heterograph(CC_graph_data2, num_nodes_dict=num_nodes_dict_p_cc)
    return MF_P_graph, BP_P_graph, CC_P_graph, P_MF_graph, P_BP_graph, P_CC_graph

def Get_weight_GOsim_graph(sim, num_node):
    GO_sim_src, GO_sim_dst, sim_value = [], [], []
    for i in range(len(sim)):
        for j in range(len(sim)):
            if sim[i][j] != 0:
                GO_sim_src.append(i)
                GO_sim_dst.append(j)
                sim_value.append(float(sim[i][j]))
    GO_src, GO_dst = torch.tensor(GO_sim_src), torch.tensor(GO_sim_dst)
    weight = torch.tensor(sim_value)
    GO_graph = dgl.graph((GO_src, GO_dst), num_nodes=num_node)
    return GO_graph, weight

def Get_DDI_PPI_G(D_D_data, P_P_data):
    Dr_Dr_1, Dr_Dr_2 = [],[]
    P_P_1, P_P_2 = [], []
    for i in range(len(D_D_data)):
        Dr_Dr_1.append(D_D_data[i][0])
        Dr_Dr_2.append(D_D_data[i][1])
    for i in range(len(P_P_data)):
        P_P_1.append(P_P_data[i][0])
        P_P_2.append(P_P_data[i][1])
    dr_dr_src, dr_dr_dst = torch.tensor(Dr_Dr_1+Dr_Dr_2), torch.tensor(Dr_Dr_2+Dr_Dr_1)
    p_p_src, p_p_dst = torch.tensor(P_P_1+P_P_2), torch.tensor(P_P_2+P_P_1)
    DDI_G = dgl.graph((dr_dr_src, dr_dr_dst))
    PPI_G = dgl.graph((p_p_src, p_p_dst))
    return DDI_G, PPI_G

def Get_Net_Graph(Dr_D_m_data, Dr_D_t_data, P_D_m_data, Dr_Dr_data, P_P_data, num_nodes_dict):
    Dr_t_D_Dr, Dr_t_D_D = [], []
    Dr_m_D_Dr, Dr_m_D_D = [], []
    P_m_D_P, P_m_D_D = [], []
    Dr_Dr_data_src, Dr_Dr_data_dst = [], []
    P_P_data_src, P_P_data_dst = [], []
    for i in range(len(Dr_D_t_data)):
        Dr_t_D_Dr.append(Dr_D_t_data[i][0])
        Dr_t_D_D.append(Dr_D_t_data[i][1])
    for i in range(len(Dr_D_m_data)):
        Dr_m_D_Dr.append(Dr_D_m_data[i][0])
        Dr_m_D_D.append(Dr_D_m_data[i][1])
    for i in range(len(P_D_m_data)):
        P_m_D_P.append(P_D_m_data[i][0])
        P_m_D_D.append(P_D_m_data[i][1])
    for i in range(len(Dr_Dr_data)):
        Dr_Dr_data_src.append(Dr_Dr_data[i][0])
        Dr_Dr_data_dst.append(Dr_Dr_data[i][1])
    for i in range(len(P_P_data)):
        P_P_data_src.append(P_P_data[i][0])
        P_P_data_dst.append(P_P_data[i][1])
    dr_d_t_src, dr_d_t_dst = torch.tensor(Dr_t_D_Dr), torch.tensor(Dr_t_D_D)
    dr_d_m_src, dr_d_m_dst = torch.tensor(Dr_m_D_Dr), torch.tensor(Dr_m_D_D)
    p_d_m_src, p_d_m_dst = torch.tensor(P_m_D_P), torch.tensor(P_m_D_D)
    DDI_src, DDI_dst = torch.tensor(Dr_Dr_data_src+Dr_Dr_data_dst), torch.tensor(Dr_Dr_data_dst+Dr_Dr_data_src)
    PPI_src, PPI_dst = torch.tensor(P_P_data_src+P_P_data_dst), torch.tensor(P_P_data_dst+P_P_data_src)
    graph_data = {('disease', 'd-t-dr', 'drug'): (dr_d_t_dst, dr_d_t_src),
                  ('disease', 'd-m-dr', 'drug'): (dr_d_m_dst, dr_d_m_src),
                  ('disease', 'd-p', 'protein'): (p_d_m_dst, p_d_m_src),
                  ('drug', 'dr-t-d', 'disease'): (dr_d_t_src, dr_d_t_dst),
                  ('drug', 'dr-m-d', 'disease'): (dr_d_m_src, dr_d_m_dst),
                  ('protein', 'p-d', 'disease'): (p_d_m_src, p_d_m_dst),
                  ('drug', 'DDI', 'drug'): (DDI_src, DDI_dst),
                  ('protein', 'PPI', 'protein'): (PPI_src, PPI_dst)}
    G = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    return G

def Get_Disease_Graph(Dr_D_m_data, Dr_D_t_data, P_D_m_data, num_nodes_dict):
    Dr_t_D_Dr, Dr_t_D_D = [], []
    Dr_m_D_Dr, Dr_m_D_D = [], []
    P_m_D_P, P_m_D_D = [], []
    for i in range(len(Dr_D_t_data)):
        Dr_t_D_Dr.append(Dr_D_t_data[i][0])
        Dr_t_D_D.append(Dr_D_t_data[i][1])
    for i in range(len(Dr_D_m_data)):
        Dr_m_D_Dr.append(Dr_D_m_data[i][0])
        Dr_m_D_D.append(Dr_D_m_data[i][1])
    for i in range(len(P_D_m_data)):
        P_m_D_P.append(P_D_m_data[i][0])
        P_m_D_D.append(P_D_m_data[i][1])
    dr_d_t_src, dr_d_t_dst = torch.tensor(Dr_t_D_Dr), torch.tensor(Dr_t_D_D)
    dr_d_m_src, dr_d_m_dst = torch.tensor(Dr_m_D_Dr), torch.tensor(Dr_m_D_D)
    p_d_m_src, p_d_m_dst = torch.tensor(P_m_D_P), torch.tensor(P_m_D_D)
    graph_data = {('drug', 'dr-t-d', 'disease'): (dr_d_t_src, dr_d_t_dst),
                  ('drug', 'dr-m-d', 'disease'): (dr_d_m_src, dr_d_m_dst),
                  ('protein', 'p-d', 'disease'): (p_d_m_src, p_d_m_dst),
                  ('disease', 'd-t-dr', 'drug'): (dr_d_t_dst, dr_d_t_src),
                  ('disease', 'd-m-dr', 'drug'): (dr_d_m_dst, dr_d_m_src),
                  ('disease', 'd-p', 'protein'): (p_d_m_dst, p_d_m_src)}
    G = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    return G

def Add_Dr_P_Net_Graph(train_Dr_P, Dr_D_m_data, Dr_D_t_data, P_D_m_data, Dr_Dr_data, P_P_data, num_nodes_dict):
    Dr_P_Dr, Dr_P_P = [], []
    Dr_t_D_Dr, Dr_t_D_D = [], []
    Dr_m_D_Dr, Dr_m_D_D = [], []
    P_m_D_P, P_m_D_D = [], []
    Dr_Dr_data_src, Dr_Dr_data_dst = [], []
    P_P_data_src, P_P_data_dst = [], []
    for i in range(len(train_Dr_P)):
        Dr_P_Dr.append(train_Dr_P[i][0])
        Dr_P_P.append(train_Dr_P[i][1])
    for i in range(len(Dr_D_t_data)):
        Dr_t_D_Dr.append(Dr_D_t_data[i][0])
        Dr_t_D_D.append(Dr_D_t_data[i][1])
    for i in range(len(Dr_D_m_data)):
        Dr_m_D_Dr.append(Dr_D_m_data[i][0])
        Dr_m_D_D.append(Dr_D_m_data[i][1])
    for i in range(len(P_D_m_data)):
        P_m_D_P.append(P_D_m_data[i][0])
        P_m_D_D.append(P_D_m_data[i][1])
    for i in range(len(Dr_Dr_data)):
        Dr_Dr_data_src.append(Dr_Dr_data[i][0])
        Dr_Dr_data_dst.append(Dr_Dr_data[i][1])
    for i in range(len(P_P_data)):
        P_P_data_src.append(P_P_data[i][0])
        P_P_data_dst.append(P_P_data[i][1])
    dr_p_src, dr_p_dst = torch.tensor(Dr_P_Dr), torch.tensor(Dr_P_P)
    dr_d_t_src, dr_d_t_dst = torch.tensor(Dr_t_D_Dr), torch.tensor(Dr_t_D_D)
    dr_d_m_src, dr_d_m_dst = torch.tensor(Dr_m_D_Dr), torch.tensor(Dr_m_D_D)
    p_d_m_src, p_d_m_dst = torch.tensor(P_m_D_P), torch.tensor(P_m_D_D)
    DDI_src, DDI_dst = torch.tensor(Dr_Dr_data_src+Dr_Dr_data_dst), torch.tensor(Dr_Dr_data_dst+Dr_Dr_data_src)
    PPI_src, PPI_dst = torch.tensor(P_P_data_src+P_P_data_dst), torch.tensor(P_P_data_dst+P_P_data_src)
    graph_data = {('drug', 'dr-p', 'protein'): (dr_p_src, dr_p_dst),
                  ('protein', 'p-dr', 'drug'): (dr_p_dst, dr_p_src),
                  ('disease', 'd-t-dr', 'drug'): (dr_d_t_dst, dr_d_t_src),
                  ('disease', 'd-m-dr', 'drug'): (dr_d_m_dst, dr_d_m_src),
                  ('disease', 'd-p', 'protein'): (p_d_m_dst, p_d_m_src),
                  ('drug', 'dr-t-d', 'disease'): (dr_d_t_src, dr_d_t_dst),
                  ('drug', 'dr-m-d', 'disease'): (dr_d_m_src, dr_d_m_dst),
                  ('protein', 'p-d', 'disease'): (p_d_m_src, p_d_m_dst),
                  ('drug', 'DDI', 'drug'): (DDI_src, DDI_dst),
                  ('protein', 'PPI', 'protein'): (PPI_src, PPI_dst)}
    G = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    return G

def Get_GO2P_Graph(P_MF_data, P_BP_data, P_CC_data, num_nodes_dict):
    MF_P, MF_GO = [], []
    BP_P, BP_GO = [], []
    CC_P, CC_GO = [], []
    for i in range(len(P_MF_data)):
        MF_P.append(P_MF_data[i][0])
        MF_GO.append(P_MF_data[i][1])
    for i in range(len(P_BP_data)):
        BP_P.append(P_BP_data[i][0])
        BP_GO.append(P_BP_data[i][1])
    for i in range(len(P_CC_data)):
        CC_P.append(P_CC_data[i][0])
        CC_GO.append(P_CC_data[i][1])
    MF_src, MF_dst = torch.tensor(MF_P), torch.tensor(MF_GO)
    BP_src, BP_dst = torch.tensor(BP_P), torch.tensor(BP_GO)
    CC_src, CC_dst = torch.tensor(CC_P), torch.tensor(CC_GO)
    data = {('MF', 'MF-p', 'protein'): (MF_dst, MF_src),
            ('BP', 'BP-p', 'protein'): (BP_dst, BP_src),
            ('CC', 'CC-p', 'protein'): (CC_dst, CC_src)}
    G = dgl.heterograph(data, num_nodes_dict=num_nodes_dict)
    return G
