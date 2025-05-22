from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from data.dataset import Block, Atom, VOCAB
from data.converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks


def extract_pdb_ligand(pdb, lig_code, chain_id, smiles, lig_idx:int=None, use_model:int=None, fragmentation_method=None):

    if pdb.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb)
    elif pdb.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb)
    else:
        raise ValueError(f"Unsupported PDB file type, {pdb}")

    list_blocks, list_indexes = [], [] 
    
    if use_model is not None:
        structure = structure[use_model]

    for chain in structure.get_chains():

        for residue in chain:
            hetero_flag, res_number, insert_code = residue.get_id()
            if hetero_flag.strip() != '' and hetero_flag == f"H_{lig_code}" and (res_number == lig_idx or lig_idx is None):
                atoms = []
                for atom in residue:
                    atoms.append(Atom(atom.get_id(), atom.get_coord(), atom.element))
                blocks = [Block(symbol=atom.element.lower(),units=[atom]) for atom in atoms]
                if fragmentation_method is not None and smiles is not None:
                    try:
                        blocks = atom_blocks_to_frag_blocks(blocks, smiles=smiles, fragmentation_method=fragmentation_method)
                    except Exception as e:
                        print(f"Could not fragment ligand {lig_code} from {pdb}. Error={e}")
                indexes = [f"{res_number}"]*len(blocks)
                list_blocks.append(blocks)
                list_indexes.append(indexes)
    if len(list_blocks) == 0:
        raise ValueError(f"Could not find ligand {lig_code} at {lig_idx} in {pdb}.")
    return list_blocks, list_indexes