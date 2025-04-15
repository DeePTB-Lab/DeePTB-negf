from .. import _keys
import ase
import numpy as np
import re
import logging
from ...constants import anglrMId
from .. import AtomicData, AtomicDataDict

log = logging.getLogger(__name__)

def block_to_feature(data, idp, blocks=False, overlap_blocks=False, orthogonal=False):
    # Hamiltonian_blocks should be a h5 group in the current version
    assert blocks != False or overlap_blocks!=False, "Both feature block and overlap blocks are not provided."
    if blocks != False:
        proto_block = blocks[list(blocks.keys())[0]][:]
    else:
        proto_block = overlap_blocks[list(overlap_blocks.keys())[0]][:]

    if blocks:
        onsite_ham = []
    if overlap_blocks and not orthogonal:
        onsite_ovp = []

    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    dtype = proto_block.dtype
    
    if isinstance(proto_block, np.ndarray):
        meta_dtype = np
    else:
        raise TypeError("Hamiltonian blocks should be np.ndarray.")

    if isinstance(data, AtomicData):
        if not hasattr(data, _keys.ATOMIC_NUMBERS_KEY):
            setattr(data, _keys.ATOMIC_NUMBERS_KEY, idp.untransform(data[_keys.ATOM_TYPE_KEY]))
    if isinstance(data, dict):
        if data.get(_keys.ATOMIC_NUMBERS_KEY, None) is None:
            data[_keys.ATOMIC_NUMBERS_KEY] = idp.untransform(data[_keys.ATOM_TYPE_KEY])
    atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY].flatten()

    # onsite features
    if blocks or overlap_blocks:
        if blocks:
            if blocks.get("0_0_0_0_0") is None:
                start_id = 1
            else:
                start_id = 0
        else:
            if overlap_blocks.get("0_0_0_0_0") is None:
                start_id = 1
            else:
                start_id = 0


        for atom in range(len(atomic_numbers)):
            block_index = '_'.join(map(str, map(int, [atom+start_id, atom+start_id] + list([0, 0, 0]))))

            if blocks:
                try:
                    block = blocks[block_index][:]
                except:
                    raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")
            
            if overlap_blocks and not orthogonal:
                try:
                    overlap_block = overlap_blocks[block_index][:]
                except:
                    raise IndexError("Overlap block for onsite not found, check Overlap file.")
                
                onsite_ovp_out = meta_dtype.zeros(idp.reduced_matrix_element)

            # if isinstance(block, torch.Tensor):
            #     block = block.cpu().detach().numpy()
            symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
            basis_list = idp.basis[symbol]
            onsite_out = meta_dtype.zeros(idp.reduced_matrix_element)

            for index, basis_i in enumerate(basis_list):
                slice_i = idp.orbital_maps[symbol][basis_i]  
                for basis_j in basis_list[index:]:
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
                    full_basis_j = idp.basis_to_full_basis[symbol][basis_j]
                    pair_ij = full_basis_i + "-" + full_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]

                    if blocks:
                        block_ij = block[slice_i, slice_j]
                        onsite_out[feature_slice] = block_ij.flatten()

                    if overlap_blocks and not orthogonal:
                        overlap_block_ij = overlap_block[slice_i, slice_j]
                        onsite_ovp_out[feature_slice] = overlap_block_ij.flatten()

            if blocks:
                onsite_ham.append(onsite_out)
            if overlap_blocks and not orthogonal:
                onsite_ovp.append(onsite_ovp_out)
        #onsite_ham = np.array(onsite_ham)

    # edge features
    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
    edge_type = idp.transform_bond(*data[_keys.ATOMIC_NUMBERS_KEY][edge_index]).flatten()
    if overlap_blocks:
        ovp_features = np.zeros((len(edge_index[0]), idp.reduced_matrix_element))
    if blocks:
        edge_features = np.zeros((len(edge_index[0]), idp.reduced_matrix_element))

    if blocks or overlap_blocks:
        for bt in range(len(idp.bond_types)):

            symbol_i, symbol_j = idp.bond_types[bt].split("-")
            basis_i = idp.basis[symbol_i]
            basis_j = idp.basis[symbol_j]
            mask = edge_type == bt
            if np.all(~mask):
                continue
            b_edge_index = edge_index[:, mask]
            b_edge_cell_shift = edge_cell_shift[mask]
            ijR = np.concatenate([b_edge_index.T+start_id, b_edge_cell_shift], axis=1).astype(np.int64).tolist()
            rev_ijR = np.concatenate([b_edge_index[[1, 0]].T+start_id, -b_edge_cell_shift], axis=1).astype(np.int64).tolist()
            ijR = list(map(lambda x: '_'.join(map(str, x)), ijR))
            rev_ijR = list(map(lambda x: '_'.join(map(str, x)), rev_ijR))

            if blocks:
                b_blocks = []
                for i,j in zip(ijR, rev_ijR):
                    if i in blocks:
                        b_blocks.append(blocks[i][:])
                    elif j in blocks:
                        b_blocks.append(blocks[j][:].T)
                    else:
                        b_blocks.append(meta_dtype.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), dtype=dtype))
                # b_blocks = [blocks[i] if i in blocks else blocks[j].T for i,j in zip(ijR, rev_ijR)]
                b_blocks = np.stack(b_blocks, axis=0)

            if overlap_blocks:
                s_blocks = []
                for i,j in zip(ijR, rev_ijR):
                    if i in overlap_blocks:
                        s_blocks.append(overlap_blocks[i][:])
                    elif j in overlap_blocks:
                        s_blocks.append(overlap_blocks[j][:].T)
                    else:
                        s_blocks.append(meta_dtype.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), dtype=dtype))
                # s_blocks = [overlap_blocks[i] if i in overlap_blocks else overlap_blocks[j].T for i,j in zip(ijR, rev_ijR)]
                s_blocks = np.stack(s_blocks, axis=0)

            for orb_i in basis_i:
                slice_i = idp.orbital_maps[symbol_i][orb_i]
                for orb_j in basis_j:
                    slice_j = idp.orbital_maps[symbol_j][orb_j]
                    full_orb_i = idp.basis_to_full_basis[symbol_i][orb_i]
                    full_orb_j = idp.basis_to_full_basis[symbol_j][orb_j]
                    if idp.full_basis.index(full_orb_i) <= idp.full_basis.index(full_orb_j):
                        pair_ij = full_orb_i + "-" + full_orb_j
                        feature_slice = idp.orbpair_maps[pair_ij]
                        if blocks:
                            edge_features[mask, feature_slice] = np.asarray(b_blocks[:,slice_i, slice_j].reshape(b_edge_index.shape[1], -1))
                        if overlap_blocks:
                            ovp_features[mask, feature_slice] = np.asarray(s_blocks[:,slice_i, slice_j].reshape(b_edge_index.shape[1], -1))

    if blocks:
        onsite_ham = np.stack(onsite_ham, axis=0)
        data[_keys.NODE_FEATURES_KEY] = np.asarray(onsite_ham)
        data[_keys.EDGE_FEATURES_KEY] = edge_features
    if overlap_blocks:
        if not orthogonal:
            onsite_ovp = np.stack(onsite_ovp, axis=0)
            data[_keys.NODE_OVERLAP_KEY] = np.asarray(onsite_ovp)
        data[_keys.EDGE_OVERLAP_KEY] = ovp_features


def feature_to_block(data, idp, overlap: bool = False):
    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    has_block = False
    if not overlap:
        if data.get(_keys.NODE_FEATURES_KEY, None) is not None:
            node_features = data[_keys.NODE_FEATURES_KEY]
            edge_features = data[_keys.EDGE_FEATURES_KEY]
            has_block = True
            blocks = {}
    else:
        if data.get(_keys.NODE_OVERLAP_KEY, None) is not None:
            node_features = data[_keys.NODE_OVERLAP_KEY]
            edge_features = data[_keys.EDGE_OVERLAP_KEY]
            has_block = True
            blocks = {}
        else:
            raise KeyError("Overlap features not found in data.")

    if has_block:
        # get node blocks from node_features
        for atom, onsite in enumerate(node_features):
            symbol = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom].reshape(-1))]
            basis_list = idp.basis[symbol]
            block = np.zeros((idp.norbs[symbol], idp.norbs[symbol]), dtype=node_features.dtype)

            for index, basis_i in enumerate(basis_list):
                f_basis_i = idp.basis_to_full_basis[symbol].get(basis_i)
                slice_i = idp.orbital_maps[symbol][basis_i]
                li = anglrMId[re.findall(r"[a-zA-Z]+", basis_i)[0]]
                for basis_j in basis_list[index:]:
                    f_basis_j = idp.basis_to_full_basis[symbol].get(basis_j)
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", basis_j)[0]]
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    pair_ij = f_basis_i + "-" + f_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    block_ij = onsite[feature_slice].reshape(2*li+1, 2*lj+1)
                    block[slice_i, slice_j] = block_ij
                    if slice_i != slice_j:
                        block[slice_j, slice_i] = block_ij.T

            block_index = '_'.join(map(str, map(int, [atom, atom] + list([0, 0, 0]))))
            blocks[block_index] = block
        
        # get edge blocks from edge_features
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
        for edge, hopping in enumerate(edge_features):
            atom_i, atom_j, R_shift = edge_index[0][edge], edge_index[1][edge], edge_cell_shift[edge]
            symbol_i = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom_i].reshape(-1))]
            symbol_j = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom_j].reshape(-1))]
            block = np.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), dtype=edge_features.dtype)

            for index, f_basis_i in enumerate(idp.full_basis):
                basis_i = idp.full_basis_to_basis[symbol_i].get(f_basis_i)
                if basis_i is None:
                    continue
                li = anglrMId[re.findall(r"[a-zA-Z]+", basis_i)[0]]
                slice_i = idp.orbital_maps[symbol_i][basis_i]
                for f_basis_j in idp.full_basis[index:]:
                    basis_j = idp.full_basis_to_basis[symbol_j].get(f_basis_j)
                    if basis_j is None:
                        continue
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", basis_j)[0]]
                    slice_j = idp.orbital_maps[symbol_j][basis_j]
                    pair_ij = f_basis_i + "-" + f_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    block_ij = hopping[feature_slice].reshape(2*li+1, 2*lj+1)
                    if f_basis_i == f_basis_j:
                        block[slice_i, slice_j] = 0.5 * block_ij
                    else:
                        block[slice_i, slice_j] = block_ij

            block_index = '_'.join(map(str, map(int, [atom_i, atom_j] + list(R_shift))))
            if atom_i < atom_j:
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[block_index] += block
            elif atom_i == atom_j:
                r_index = '_'.join(map(str, map(int, [atom_i, atom_j] + list(-R_shift))))
                if blocks.get(r_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[r_index] += block.T
            else:
                block_index = '_'.join(map(str, map(int, [atom_j, atom_i] + list(-R_shift))))
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block.T
                else:
                    blocks[block_index] += block.T
                    
    return blocks