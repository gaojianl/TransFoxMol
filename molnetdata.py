import networkx as nx
import numpy as np
import pandas as pd
import os
import argparse
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from TFM.Dataset import MolNet
remover = SaltRemover()
smile_graph = {}
meta = ['W', 'U', 'Zr', 'He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']


def atom_features(atom):
    res = one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','F','P','S','Cl','Br','I','B','Si','Unknown']) + \
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3, 4, 5, 6]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + [atom.GetIsAromatic()] + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1,0,1,3]) + \
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])
    try:
        res = res + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        bonds = atom.GetBonds()
        for bond in bonds:
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and str(bond.GetStereo()) in ["STEREOZ", "STEREOE"]:
                res = res + one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREOZ", "STEREOE"]) + [atom.HasProp('_ChiralityPossible')]
        if len(res) == 33:
            res = res + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return np.array(res)


def order_gnn_features(bond):
    weight = [1, 2, 3, 1.5]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]

    for i, m in enumerate(bond_feats):
        if m == True and i != 0:
            b = weight[i]
        elif m == True and i == 0:
            if bond.GetIsConjugated() == True:
                b = 1.4
            else:
                b = 1
        else:pass
    return b           


def order_tf_features(bond):
    weight = [1, 2, 3, 1.5]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
    for i, m in enumerate(bond_feats):
        if m == True:
            b = weight[i]
    return b        


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiletopyg(smi):
    g = nx.Graph()
    mol = Chem.MolFromSmiles(smi)
    c_size = mol.GetNumAtoms()

    features = []
    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        feature = atom_features(atom)
        features.append((feature / sum(feature)).tolist()) 

    c = []
    adj_order_matrix = np.eye(c_size)
    dis_order_matrix = np.zeros((c_size,c_size))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bfeat = order_gnn_features(bond)
        g.add_edge(a1, a2, weight=bfeat)
        tfft = order_tf_features(bond)
        adj_order_matrix[a1, a2] = tfft
        adj_order_matrix[a2, a1] = tfft
        if bond.GetIsConjugated():
            c = list(set(c).union(set([a1, a2])))

    g = g.to_directed()
    edge_index = np.array(g.edges).tolist()

    edge_attr = []
    for w in list(g.edges.data('weight')):
        edge_attr.append(w[2])

    for i in range(c_size):
        for j in range(i,c_size):
            if adj_order_matrix[i, j] == 0 and i != j:
                conj = False
                paths = list(nx.node_disjoint_paths(g, i, j))
                if len(paths) > 1: 
                    paths = sorted(paths, key=lambda i:len(i),reverse=False)
                for path in paths:
                    if set(path) < set(c):
                        conj = True
                        break
                if conj:
                    adj_order_matrix[i, j] = 1.1
                    adj_order_matrix[j, i] = 1.1
                else:
                    path = paths[0]
                    dis_order_matrix[i, j] = len(path) - 1
                    dis_order_matrix[j, i] = len(path) - 1

    g = [c_size, features, edge_index, edge_attr, adj_order_matrix, dis_order_matrix]
    return [smi, g]


def write(res):
    smi, g = res
    smile_graph[smi] = g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransFoxMol')
    parser.add_argument('--moldata', type=str, help='dataset name to process')
    parser.add_argument('--task', type=str, choices=['clas', 'reg'], help='classification or regression')
    parser.add_argument('--ncpu', type=int, default=4, help='number of cpus to use (default: 4)')
    args = parser.parse_args()

    moldata = args.moldata
    if moldata in ['esol', 'freesolv', 'lipo', 'qm7']:
        task = 'reg'
        numtasks = 1
        if moldata == 'esol':
            labell = ['measured log solubility in mols per litre']
        elif moldata == 'freesolv':
            labell = ['expt']
        elif moldata == 'lipo':
            labell = ['exp']
        elif moldata == 'qm7':
            labell = ['u0_atom']
        else:
            labell = ['standard_value']

    elif moldata in ['bbbp', 'sider', 'clintox', 'tox21', 'bace']:
        task = 'clas'
        if moldata == 'sider':
            numtasks = 27
            labell = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders','Investigations', 'Musculoskeletal and connective tissue disorders', 
                      'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders','Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                      'General disorders and administration site conditions','Endocrine disorders', 'Surgical and medical procedures','Vascular disorders', 'Blood and lymphatic system disorders',
                      'Skin and subcutaneous tissue disorders','Congenital, familial and genetic disorders','Infections and infestations','Respiratory, thoracic and mediastinal disorders','Psychiatric disorders', 
                      'Renal and urinary disorders','Pregnancy, puerperium and perinatal conditions','Ear and labyrinth disorders', 'Cardiac disorders','Nervous system disorders','Injury, poisoning and procedural complications']
        elif moldata == 'clintox':
            numtasks = 2
            labell = ['FDA_APPROVED', 'CT_TOX']
        elif moldata == 'tox21':
            numtasks = 12
            labell = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        elif moldata == 'bace':
            numtasks = 1
            labell = ['Class']
        elif moldata == 'bbbp':
            numtasks = 1
            labell = ['p_np']
        else:
            numtasks = 1
            labell = ['label']

    processed_data_file = 'dataset/processed/' + moldata+task + '_pyg.pt'
    if not os.path.isfile(processed_data_file):
        try:
            df = pd.read_csv('./dataset/raw/'+moldata+'.csv')
        except:
            print('Raw data not found! Put the right raw csvfile in **/dataset/raw/')
        compound_iso_smiles = np.array(df['smiles']) 
        ic50s = np.array(df[labell])
        #ic50s = -np.log10(np.array(ic50s))
        pool = Pool(args.ncpu)
        smis = []
        y = []
        result = []

        for smi, label in zip(compound_iso_smiles, ic50s):
            smis.append(smi) 
            y.append(label)             
            result.append(pool.apply_async(smiletopyg, (smi,)))
        pool.close()
        pool.join()

        for res in result:
            smi, g = res.get()
            smile_graph[smi] = g

        MolNet(root='./dataset', dataset=moldata+task, xd=smis, y=y, smile_graph=smile_graph)
    