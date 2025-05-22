import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.converter.pdb_lig_to_blocks import extract_pdb_ligand
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.dataset import blocks_interface, blocks_to_data

def parse_args():
    parser = argparse.ArgumentParser(description='Process PDB data for embedding with ATOMICA')
    parser.add_argument('--data_index_file', type=str, default='/blue/yanjun.li/share/GatorMol/GatorMol-dev/ATOMICA/LP_PDBBind/LP_PDBBind_train/apo_ligand.csv')
    parser.add_argument('--out_path', type=str, default='/blue/yanjun.li/share/GatorMol/GatorMol-dev/ATOMICA/LP_PDBBind/LP_PDBBind_train/apo_ligand.pkl', help='Output path')
    parser.add_argument('--fragmentation_method', type=str, default='PS_300', choices=['PS_300'], help='fragmentation method for small molecule ligands')
    return parser.parse_args()

def process_PL_pdb(input_type,
                   pdb_file,
                   pdb_id,
                   rec_chain,
                   lig_code,
                   lig_chain,
                   smiles,
                   lig_resi,
                   fragmentation_method=None):
    items = []

    if input_type == 'ligand':
        # 提取配体块和对应的 PDB 序号
        list_lig_blocks, list_lig_indexes = extract_pdb_ligand(
            pdb_file,
            lig_code,
            lig_chain,
            smiles,
            lig_idx=lig_resi,
            use_model=0,
            fragmentation_method=fragmentation_method
        )

        # 对每一段配体片段单独打包
        for idx, (lig_blocks, lig_indexes) in enumerate(
                zip(list_lig_blocks, list_lig_indexes)):
            # 将这一段片段转换成 data
            data = blocks_to_data(lig_blocks)
            # 构造唯一 ID
            _id = f"{pdb_id}_{lig_chain}_{lig_code}"
            if len(list_lig_blocks) > 1:
                _id = f"{_id}_{idx}"

            # 将块索引映射到 PDB 序号，块编号从 1 开始
            block_to_pdb = {
                blk_idx + 1: pdb_idx
                for blk_idx, pdb_idx in enumerate(lig_indexes)
            }
            items.append({
                'data': data,
                'block_to_pdb_indexes': block_to_pdb,
                'id': _id,
            })

    elif input_type == 'pocket':
        # 提取受体（口袋）所有残基块和对应的 PDB 序号
        rec_blocks, rec_indexes = pdb_to_list_blocks(
            pdb_file,
            selected_chains=rec_chain,
            return_indexes=True
        )
        # 展平多链的列表
        rec_blocks_flat = sum(rec_blocks, [])
        rec_indexes_flat = sum(rec_indexes, [])
        # 整体 pocket 转成 data
        data = blocks_to_data(rec_blocks_flat)
        # 构造唯一 ID
        _id = f"{pdb_id}_{''.join(rec_chain)}"
        # 将块索引映射到 PDB 序号，块编号从 1 开始
        block_to_pdb = {
            blk_idx + 1: pdb_idx
            for blk_idx, pdb_idx in enumerate(rec_indexes_flat)
        }
        items.append({
            'data': data,
            'block_to_pdb_indexes': block_to_pdb,
            'id': _id,
        })
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return items

def group_chains(list_chain_blocks, list_chain_pdb_indexes, group1, group2):
    group1_chains = []
    group2_chains = []
    group1_indexes = []
    group2_indexes = []
    for chain_blocks, chain_pdb_indexes in zip(list_chain_blocks, list_chain_pdb_indexes):
        if chain_pdb_indexes[0].split("_")[0] in group1:
            group1_chains.extend(chain_blocks)
            group1_indexes.extend(chain_pdb_indexes)
        elif chain_pdb_indexes[0].split("_")[0] in group2:
            group2_chains.extend(chain_blocks)
            group2_indexes.extend(chain_pdb_indexes)
    return [group1_chains, group2_chains], [group1_indexes, group2_indexes]

def process_pdb(pdb_file, pdb_id, group1_chains, group2_chains, dist_th):
    blocks, pdb_indexes = pdb_to_list_blocks(pdb_file, selected_chains=group1_chains+group2_chains, return_indexes=True, use_model=0)
    if len(blocks) != 2:
        blocks, pdb_indexes = group_chains(blocks, pdb_indexes, group1_chains, group2_chains)
    blocks1, blocks2, block1_indexes, block2_indexes = blocks_interface(blocks[0], blocks[1], dist_th, return_indexes=True)
    if len(blocks1) == 0 or len(blocks2) == 0:
        return None
    pdb_indexes_map = {}
    pdb_indexes_map.update(dict(zip(range(1,len(blocks1)+1), [pdb_indexes[0][i] for i in block1_indexes])))# map block index to pdb index, +1 for global block)
    pdb_indexes_map.update(dict(zip(range(len(blocks1)+2,len(blocks1)+len(blocks2)+2), [pdb_indexes[1][i] for i in block2_indexes])))# map block index to pdb index, +1 for global block)
    data = blocks_to_data(blocks1, blocks2)
    return {
        "data": data,
        "id": f"{pdb_id}_{''.join(group1_chains)}_{''.join(group2_chains)}",
        "block_to_pdb_indexes": pdb_indexes_map,
    }

def main(args):
    data_index_file = pd.read_csv(args.data_index_file)
    items = []
    for _, row in tqdm(data_index_file.iterrows(), total=len(data_index_file)):
        input_type = row['input_type']
        pdb_file = row['pdb_file']
        pdb_id = row['pdb_id']
        chain1 = row['chain1']
        chain2 = row['chain2']
        lig_code = row['lig_code']
        smiles = row['smiles']
        lig_resi = int(row['lig_resi']) if not pd.isna(row['lig_resi']) else None
        chain1 = chain1.split("_")if not pd.isna(row['chain1']) else None
        chain2 = chain2.split("_")[0] if not pd.isna(row['chain2']) else None

        #chain2 = chain2[0]
        pl_items = process_PL_pdb(input_type=input_type,pdb_file=pdb_file, pdb_id=pdb_id, rec_chain=chain1, lig_code=lig_code, lig_chain=chain2, smiles=smiles, lig_resi=lig_resi, fragmentation_method=args.fragmentation_method)

        items.extend(pl_items)
    
    with open(args.out_path, 'wb') as f:
        pickle.dump(items, f)



if __name__ == "__main__":
    main(parse_args())
