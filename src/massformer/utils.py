import importlib
import massformer.algos1
import massformer.algos2
import numpy as np
import torch as th
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from contextlib import contextmanager

EPS = np.finfo(np.float32).eps
CHARGE_FACTOR_MAP = {
    1: 1.00,
    2: 0.90,
    3: 0.85,
    4: 0.80,
    5: 0.75,
    "large": 0.75
}
ELEMENT_LIST = ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F']

#-----------------------------gf data utils-----------------------------
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except BaseException:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]


def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def mol2graph(mol, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 


def graph2data(graph):
    """ taken from process() in https://github.com/snap-stanford/ogb/blob/master/ogb/lsc/pcqm4mv2_pyg.py """
    data = Data()
    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    # data.__num_nodes__ = int(graph['num_nodes'])
    data.num_nodes = int(graph['num_nodes'])
    data.edge_index = th.from_numpy(graph['edge_index']).to(th.int64)
    data.edge_attr = th.from_numpy(graph['edge_feat']).to(th.int64)
    data.x = th.from_numpy(graph['node_feat']).to(th.int64)
    data.y = th.Tensor([-1])  # dummy
    return data

@th.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + th.arange(0,
                                      feature_num * offset,
                                      offset,
                                      dtype=th.long)
    x = x + feature_offset
    return x


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


#-----------------------------misc utils-----------------------------

def np_one_hot(input, num_classes=None):
    """ numpy wrapper for one_hot """

    th_input = th.as_tensor(input, device="cpu")
    th_oh = th.nn.functional.one_hot(th_input, num_classes=num_classes)
    oh = th_oh.numpy()
    return oh


@contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def flatten_lol(list_of_list):
    flat_list = []
    for ll in list_of_list:
        flat_list.extend(ll)
    return flat_list

def rdkit_import(*module_strs):

    RDLogger = importlib.import_module("rdkit.RDLogger")
    RDLogger.DisableLog('rdApp.*')
    modules = []
    for module_str in module_strs:
        modules.append(importlib.import_module(module_str))
    return tuple(modules)

#-----------------------------data utils-----------------------------

def atom_type_one_hot(atom):

    chemutils = importlib.import_module("dgllife.utils")
    return chemutils.atom_type_one_hot(
        atom, allowable_set=ELEMENT_LIST, encode_unknown=True
    )


def atom_bond_type_one_hot(atom):

    chemutils = importlib.import_module("dgllife.utils")
    bs = atom.GetBonds()
    if not bs:
        return [False, False, False, False]
    bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
    return [any(bt[:, i]) for i in range(bt.shape[1])]


def mol_to_charge(mol):
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        charge = rdmolops.GetFormalCharge(mol)
    except BaseException:
        charge = np.nan
    return charge

def mol_to_smiles(
        mol,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=False):
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        smiles = Chem.MolToSmiles(
            mol,
            canonical=canonical,
            isomericSmiles=isomericSmiles,
            kekuleSmiles=kekuleSmiles)
    except BaseException:
        smiles = np.nan
    return smiles


def mol_to_formula(mol):
    modules = rdkit_import("rdkit.Chem.AllChem")
    AllChem = modules[0]
    try:
        formula = AllChem.CalcMolFormula(mol)
    except BaseException:
        formula = np.nan
    return formula

def get_charge(prec_type_str):

    if prec_type_str == "EI":
        return 1
    end_brac_idx = prec_type_str.index("]")
    charge_str = prec_type_str[end_brac_idx + 1:]
    if charge_str == "-":
        charge_str = "1-"
    elif charge_str == "+":
        charge_str = "1+"
    assert len(charge_str) >= 2
    sign = charge_str[-1]
    assert sign in ["+", "-"]
    magnitude = int(charge_str[:-1])
    if sign == "+":
        charge = magnitude
    else:
        charge = -magnitude
    return charge

def nce_to_ace_helper(nce, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    ace = (nce * prec_mz * charge_factor) / 500.
    return ace

def ace_to_nce_helper(ace, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    nce = (ace * 500.) / (prec_mz * charge_factor)
    return nce


def nce_to_ace(row):

    prec_mz = row["prec_mz"]
    nce = row["nce"]
    prec_type = row["prec_type"]
    charge = np.abs(get_charge(prec_type))
    ace = nce_to_ace_helper(nce, charge, prec_mz)
    return ace


def ace_to_nce(row):

    prec_mz = row["prec_mz"]
    ace = row["ace"]
    prec_type = row["prec_type"]
    charge = np.abs(get_charge(prec_type))
    nce = ace_to_nce_helper(ace, charge, prec_mz)
    return nce

#-----------------------------spec utils-----------------------------

def bin_func(mzs, ints, mz_max, mz_bin_res, ints_thresh, return_index):

    mzs = np.array(mzs, dtype=np.float32)
    bins = np.arange(
        mz_bin_res,
        mz_max +
        mz_bin_res,
        step=mz_bin_res).astype(
        np.float32)
    bin_idx = np.searchsorted(bins, mzs, side="right")
    if return_index:
        return bin_idx.tolist()
    else:
        ints = np.array(ints, dtype=np.float32)
        bin_spec = np.zeros([len(bins)], dtype=np.float32)
        for i in range(len(mzs)):
            if bin_idx[i] < len(bin_spec) and ints[i] >= ints_thresh:
                bin_spec[bin_idx[i]] = max(bin_spec[bin_idx[i]], ints[i])
        if np.all(bin_spec == 0.):
            print("> warning: bin_spec is all zeros!")
            bin_spec[-1] = 1.
        return bin_spec
    
def process_spec(spec, transform, normalization, eps=EPS):

    # scale spectrum so that max value is 1000
    spec = spec / (th.max(spec, dim=-1, keepdim=True)[0] + eps) * 1000.
    # transform signal
    if transform == "log10":
        spec = th.log10(spec + 1)
    elif transform == "log10over3":
        spec = th.log10(spec + 1) / 3
    elif transform == "loge":
        spec = th.log(spec + 1)
    elif transform == "sqrt":
        spec = th.sqrt(spec)
    elif transform == "linear":
        raise NotImplementedError
    elif transform == "none":
        pass
    else:
        raise ValueError("invalid transform")
    # normalize
    if normalization == "l1":
        spec = F.normalize(spec, p=1, dim=-1, eps=eps)
    elif normalization == "l2":
        spec = F.normalize(spec, p=2, dim=-1, eps=eps)
    elif normalization == "none":
        pass
    else:
        raise ValueError("invalid normalization")
    assert not th.isnan(spec).any()
    return spec


def process_spec_old(spec, transform, normalization, ints_thresh):

    # scale spectrum so that max value is 1000
    spec = spec * (1000. / np.max(spec))
    # remove noise
    spec = spec * (spec > ints_thresh * np.max(spec)).astype(float)
    # transform signal
    if transform == "log10":
        spec = np.log10(spec + 1)
    elif transform == "log10over3":
        spec = np.log10(spec + 1) / 3
    elif transform == "loge":
        spec = np.log(spec + 1)
    elif transform == "sqrt":
        spec = np.sqrt(spec)
    elif transform == "linear":
        raise NotImplementedError
    elif transform == "none":
        pass
    else:
        raise ValueError("invalid transform")
    # normalize
    if normalization == "l1":
        spec = spec / np.sum(np.abs(spec))
    elif normalization == "l2":
        spec = spec / np.sqrt(np.sum(spec**2))
    elif normalization == "none":
        pass
    else:
        raise ValueError("invalid spectrum_normalization")
    return spec


def unprocess_spec(spec, transform):

    # transform signal
    if transform == "log10":
        max_ints = float(np.log10(1000. + 1.))
        def untransform_fn(x): return 10**x - 1.
    elif transform == "log10over3":
        max_ints = float(np.log10(1000. + 1.) / 3.)
        def untransform_fn(x): return 10**(3 * x) - 1.
    elif transform == "loge":
        max_ints = float(np.log(1000. + 1.))
        def untransform_fn(x): return th.exp(x) - 1.
    elif transform == "sqrt":
        max_ints = float(np.sqrt(1000.))
        def untransform_fn(x): return x**2
    elif transform == "linear":
        raise NotImplementedError
    elif transform == "none":
        max_ints = 1000.
        def untransform_fn(x): return x
    else:
        raise ValueError("invalid transform")
    spec = spec / (th.max(spec, dim=-1, keepdim=True)[0] + EPS) * max_ints
    spec = untransform_fn(spec)
    spec = th.clamp(spec, min=0.)
    assert not th.isnan(spec).any()
    return spec