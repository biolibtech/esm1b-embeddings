import time
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from hashlib import md5
import os
from tqdm import tqdm
import numpy as np
import json
import argparse
import io
from Bio import SeqIO
import pandas as pd
import csv  


os.mkdir('output')

parser = argparse.ArgumentParser()
parser.add_argument('--fasta')
args = parser.parse_args()

input_str = open(args.fasta, "r").read()

if not input_str.strip().startswith(">") and input_str.strip() != "":
    input_str = "> Unnamed Protein \n" + input_str

input_str_buffer = io.StringIO(input_str)

sequences = []
for idx, record in enumerate(SeqIO.parse(input_str_buffer, "fasta")):
    sequences.append(str(record.seq))

fasta_len = len(sequences)
chunk_size = 1
repr_layers = 33
print("Step 1/2 | Loading transformer model...")
esm_args = torch.load('esm_model_args.pt')
esm_alphabet = torch.load('esm_model_alphabet.pt')
esm_model_state_dict = torch.load('esm_model_state_dict.pt')

esm_model = ProteinBertModel(
    args=esm_args,
    alphabet=esm_alphabet
)

esm_model.load_state_dict(esm_model_state_dict)

print("\nStep 2/2 | Generating embeddings for sequences...")
with torch.no_grad():
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')
        esm_model = esm_model.cuda()
    esm_model.eval()
    batch_converter = esm_alphabet.get_batch_converter()
    
    mapping_dict = {}
    
    for idx, seq in enumerate(tqdm(sequences, unit='seq', desc='Generating embeddings')):
        
        mapping_dict[seq] = idx
        
        seqs = list([("seq", s) for s in [seq]])
        labels, strs, toks = batch_converter(seqs)
        repr_layers_list = [
            (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
        ]

        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)

        minibatch_max_length = toks.size(1)

        tokens_list = []
        end = 0
        while end <= minibatch_max_length:
            start = end
            end = start + 1022
            if end <= minibatch_max_length:
                # we are not on the last one, so make this shorter
                end = end - 300
            tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)[
                "representations"][repr_layers - 1]
            tokens_list.append(tokens)

        out = torch.cat(tokens_list, dim=1)

        # set nan to zeros
        out[out != out] = 0.0

        res = out.transpose(0, 1)[1:-1]
        seq_embedding = res[:, 0].detach().cpu().numpy()
        
        np.savez(f'output/{idx}', fix_imports=True, allow_pickle=False, embedding=seq_embedding)
        
    json_file_handle = open('output/embedding_map.json', 'w')
    json.dump(mapping_dict, json_file_handle)