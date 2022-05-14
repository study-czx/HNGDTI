import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import torch.nn.functional as F
import funcs
from dgl.nn.pytorch import HeteroGraphConv as HGCN
from dgl.nn.pytorch import SAGEConv as SAGE

funcs.setup_seed(1)

data_types = ["DTI-rand/", "DTI-net/"]
types = ["random", "new_drug", "new_protein", "new_drug_protein"]
# neg3,5,7,9 and neg3-b,neg5-b,neg7-b,neg9-b
# data_types = ["neg3_9/", "neg-b3_9/"]
# types = ["neg3", "neg5", "neg7", "neg9"]

b_size, n_hidden = 128, 128
lr, wd = 1e-4, 1e-4
num_epoches = 100

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

Drug_id = np.loadtxt(r"./network_node/All_drug_id_2223.csv", dtype=object, delimiter=",", skiprows=1)
Protein_id = np.loadtxt(r"./network_node/All_protein_id_13816.csv", dtype=object, delimiter=",", skiprows=1)
Disease_id = np.loadtxt(r"./network_node/All_disease_id_7061.csv", dtype=object, delimiter=",", skiprows=1)

n_drugs, n_proteins, n_diseases = len(Drug_id), len(Protein_id), len(Disease_id)

Pubchem = np.loadtxt(r"./feature/Pubchem.csv", dtype=float, delimiter=",", skiprows=0)
KSCTriad = np.loadtxt(r"./feature/KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)

Dr_D_m = np.loadtxt(r"./network/Dr_D_m_39187.csv", dtype=object, delimiter=",", skiprows=1)
Dr_D_t = np.loadtxt(r"./network/Dr_D_t_21908.csv", dtype=object, delimiter=",", skiprows=1)
P_D_m = np.loadtxt(r"./network/P_D_m_29201.csv", dtype=object, delimiter=",", skiprows=1)

dr_id_map, p_id_map, d_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id), funcs.id_map(Disease_id)

Dr_D_m_data = funcs.Get_index(Dr_D_m, dr_id_map, d_id_map)
Dr_D_t_data = funcs.Get_index(Dr_D_t, dr_id_map, d_id_map)
P_D_m_data = funcs.Get_index(P_D_m, p_id_map, d_id_map)

num_nodes_dict = {'drug': n_drugs, 'disease': n_diseases, 'protein': n_proteins}
my_G = funcs.Get_Disease_Graph(Dr_D_m_data, Dr_D_t_data, P_D_m_data, num_nodes_dict)

my_G = my_G.to(device)

finger_feats = Pubchem
seq_feats = KSCTriad

finger_feats = torch.as_tensor(torch.from_numpy(finger_feats), dtype=torch.float32).to(device)
seq_feats = torch.as_tensor(torch.from_numpy(seq_feats), dtype=torch.float32).to(device)
disease_feats = torch.as_tensor(torch.from_numpy(np.identity(n_diseases)), dtype=torch.float32).to(device)

n_diseases_feature = len(disease_feats[0])
n_finger_feature, n_seq_feature = len(finger_feats[0]), len(seq_feats[0])

print("Disease_feature_length:", n_diseases_feature)
print("Drug_finger_length:", n_finger_feature, "Protein_seq_length:", n_seq_feature)

class Dr_P_Embedding(nn.Module):
    def __init__(self):
        super(Dr_P_Embedding, self).__init__()
        self.drug_embedding = nn.Sequential(nn.Linear(in_features=n_finger_feature, out_features=n_hidden), nn.ReLU())
        self.protein_embedding = nn.Sequential(nn.Linear(in_features=n_seq_feature, out_features=n_hidden), nn.ReLU())

    def forward(self, Drug_feature, Protein_feature):
        h_dr = self.drug_embedding(Drug_feature)
        h_p = self.protein_embedding(Protein_feature)
        return h_dr, h_p

class Other_Embedding(nn.Module):
    def __init__(self):
        super(Other_Embedding, self).__init__()
        self.d_embedding = nn.Sequential(nn.Linear(in_features=n_diseases_feature, out_features=n_hidden), nn.ReLU())

    def forward(self, Disease_feature):
        h_d = self.d_embedding(Disease_feature)
        return h_d

drop_feat = 0
agg = "pool"
class All_Graph_Net(nn.Module):
    def __init__(self):
        super(All_Graph_Net, self).__init__()
        self.Graph_Net = HGCN(
            {'d-t-dr': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden,  feat_drop=drop_feat, aggregator_type=agg, activation=F.relu),
             'd-m-dr': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg, activation=F.relu),
             'd-p': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg, activation=F.relu),
             'dr-t-d': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg, activation=F.relu),
             'dr-m-d': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg, activation=F.relu),
             'p-d': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg, activation=F.relu)},
            aggregate='sum')

    def forward(self, h_dr, h_p, h_d, my_G):
        h = {'drug': h_dr, 'protein': h_p, 'disease':h_d}
        h1 = self.Graph_Net(my_G, h)
        h2 = self.Graph_Net(my_G, h1)
        # h3 = self.Graph_Net(my_G, h2)
        h_dr1, h_p1 = h1['drug'], h1['protein']
        h_dr2, h_p2 = h2['drug'], h2['protein']
        # h_dr3, h_p3 = h3['drug'], h3['protein']
        return h_dr1, h_p1, h_dr2, h_p2


class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.dr_p_embedding = Dr_P_Embedding()
        self.other_embedding = Other_Embedding()
        self.All_Net = All_Graph_Net()
        self.connected_layer1 = nn.Sequential(nn.Linear(in_features=n_hidden*6, out_features=n_hidden*2), nn.BatchNorm1d(num_features=n_hidden*2),nn.ReLU())
        self.connected_layer2 = nn.Sequential(nn.Linear(in_features=n_hidden*2, out_features=n_hidden), nn.BatchNorm1d(num_features=n_hidden),nn.ReLU())
        self.connected_layer3 = nn.Sequential(nn.Linear(in_features=n_hidden, out_features=64), nn.BatchNorm1d(num_features=64),nn.ReLU())
        self.output = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()
    # 前向传播
    def forward(self, x_dr, x_p, finger_feats, seq_feats,  disease_feat, my_G):
        h_dr_finger, h_p_seq = self.dr_p_embedding(finger_feats, seq_feats)
        h_d = self.other_embedding(disease_feat)
        h_dr1, h_p1, h_dr2, h_p2 = self.All_Net(h_dr_finger, h_p_seq, h_d, my_G)
        dr_new = torch.cat((h_dr_finger, h_dr1, h_dr2), dim=1)
        p_new = torch.cat((h_p_seq, h_p1, h_p2), dim=1)
        dr_feat, p_feat = dr_new[x_dr].squeeze(1), p_new[x_p].squeeze(1)
        h_dr_p = torch.cat((dr_feat, p_feat), dim=1)
        out = self.connected_layer1(h_dr_p)
        out = self.connected_layer2(out)
        out = self.connected_layer3(out)
        out = self.sigmoid(self.output(out))
        return out

types = ["random", "new_drug", "new_protein", "new_drug_protein"]
for data_type in data_types:
    print("data_type: ", data_type)
    for type in types:
        print("type: ", type)
        output_score = np.zeros(shape=(4, 5))
        for i in range(5):
            seed_type = "seed" + str(i + 1)
            train_P = np.loadtxt(data_type + seed_type + "/" + type + "/train_P.csv", dtype=str, delimiter=",",
                                 skiprows=1)
            dev_P = np.loadtxt(data_type + seed_type + "/" + type + "/dev_P.csv", dtype=str, delimiter=",", skiprows=1)
            test_P = np.loadtxt(data_type + seed_type + "/" + type + "/test_P.csv", dtype=str, delimiter=",",
                                skiprows=1)
            train_N = np.loadtxt(data_type + seed_type + "/" + type + "/train_N.csv", dtype=str, delimiter=",",
                                 skiprows=1)
            dev_N = np.loadtxt(data_type + seed_type + "/" + type + "/dev_N.csv", dtype=str, delimiter=",", skiprows=1)
            test_N = np.loadtxt(data_type + seed_type + "/" + type + "/test_N.csv", dtype=str, delimiter=",",
                                skiprows=1)
            print("number of DTI: ", len(train_P), len(dev_P), len(test_P))
            print("number of Negative DTI ", len(train_N), len(dev_N), len(test_N))
            train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map, p_id_map)
            dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map, p_id_map)
            test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map, p_id_map)
            train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
            dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
            test_loader = funcs.get_test_loader(test_X, test_Y, b_size)
            best_auc, best_epoch, best_extra = 0, 0, 0
            best_test = []
            losses = nn.BCELoss()
            model = My_Net().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=20,
                                                                   verbose=False)
            for epoch in range(num_epoches):
                train_loss = 0
                train_scores, train_scores_label, train_labels = [], [], []
                for step, (batch_x, batch_y) in enumerate(train_loader):
                    model.train()
                    b_x = batch_x.long().to(device)
                    b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                    b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                    b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                    output = model(b_x_dr, b_x_p, finger_feats, seq_feats, disease_feats, my_G)
                    score = torch.squeeze(output, dim=1)
                    loss = losses(score, b_y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                    train_scores = np.concatenate((train_scores, scores))
                    train_labels = np.concatenate((train_labels, label))
                train_scores_label = funcs.computer_label(train_scores, 0.5)
                train_avloss = train_loss / len(train_loader)
                train_acc = skm.accuracy_score(train_labels, train_scores_label)
                train_auc = skm.roc_auc_score(train_labels, train_scores)

                dev_scores, dev_labels = [], []
                test_scores, test_scores_label, test_labels = [], [], []
                extra_scores, extra_scores_label, extra_labels = [], [], []
                with torch.no_grad():
                    for step, (batch_x, batch_y) in enumerate(dev_loader):
                        model.eval()
                        b_x = batch_x.long().to(device)
                        b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = model(b_x_dr, b_x_p, finger_feats, seq_feats, disease_feats, my_G)
                        score = torch.squeeze(output, dim=1)
                        scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                        dev_scores = np.concatenate((dev_scores, scores))
                        dev_labels = np.concatenate((dev_labels, label))
                    dev_auc = skm.roc_auc_score(dev_labels, dev_scores)
                    scheduler.step(dev_auc)
                    # 测试集
                    for step, (batch_x, batch_y) in enumerate(test_loader):
                        model.eval()
                        b_x = batch_x.long().to(device)
                        b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = model(b_x_dr, b_x_p, finger_feats, seq_feats, disease_feats, my_G)
                        score = torch.squeeze(output, dim=1)
                        scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                        test_scores = np.concatenate((test_scores, scores))
                        test_labels = np.concatenate((test_labels, label))
                    test_scores_label = funcs.computer_label(test_scores, 0.5)
                    test_acc = skm.accuracy_score(test_labels, test_scores_label)
                    test_auc = skm.roc_auc_score(test_labels, test_scores)
                    test_aupr = skm.average_precision_score(test_labels, test_scores)
                    # 额外测试集

                print(
                    'epoch:{},Train Loss: {:.4f},Train Acc: {:.4f},Train Auc: {:.4f},Dev Auc: {:.4f}, Test Acc: {:.4f},Test Auc: {:.4f},TestAUPR: {:.4f}'
                        .format(epoch, train_avloss, train_acc, train_auc, dev_auc, test_acc, test_auc, test_aupr))
                if dev_auc >= best_auc:
                    best_auc = dev_auc
                    best_epoch = epoch
                    best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f')]
            print("best_dev_AUC:", best_auc)
            print("best_epoch", best_epoch)
            print("test_out", best_test)
            output_score[0][i], output_score[1][i], output_score[2][i], output_score[3][i] = best_auc, best_test[0], \
                                                                                             best_test[1], best_test[2]

        print(output_score)
        mean_acc, mean_auc, mean_mcc = np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(
            output_score[3])
        std_acc, std_auc, std_mcc = np.nanstd(output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3])
        print(mean_acc, mean_auc, mean_mcc)
        print(std_acc, std_auc, std_mcc)
