from collections import Counter
from pathlib import Path
from typing import Dict

from pyfaidx import Fasta


def csv2dict(infile: Path) -> Dict[str, str]:
    """
    Simple function to read the a csv into a dictionary (faster than pandas)
    Args:
        infile: Path to input file, comma delimited

    Returns:dictionary with first field as key and second as value
    """
    dictionary = {}
    with open(infile) as inf:
        for line in inf:
            if line:
                key, value = line.strip().split(',')
                dictionary[key] = value
    return dictionary


def preprocessing(data_set: Path, max_clust_size: int, metadata: Path):
    """
    TODO: update these doc strings
    Preprocessing of fasta sequences using BioPython into a database of
    SeqRecord objects each representing a unique sequence, can handle
    multiple sequence fastas.

    seqs: main sequence database, dictionary-like object, no info on
    clusters cluster_names: list of all cluster names from sub-directory
     names.

    number_of_clusters: integer of the total count of clusters.

    cluster_sample_info: Dictionary with keys=cluster_names and values =
    a tuple consisting of: (number of samples in cluster,
        a list of accession ids corresponding to sequences of that cluster).

    total_seq: integer of the total sequence count in dataset.

    """
    # Iterate through all fasta files and read it into data structure
    outfn = data_set.joinpath('all_seqs.fasta').resolve()
    cluster_dict = csv2dict(metadata)
    cluster_stats = Counter(cluster_dict.values())
    if outfn.exists():
        print(f'File {outfn} exists and will be used! If this is '
              f'unintended, please remove the file')
    else:
        for file in data_set.glob('*'):
            if file.suffix != '.fai':
                with open(file) as infile, open(outfn, 'a') as outfile:
                    outfile.write(f'{infile.read().strip()}\n')
    seq_dict = Fasta(outfn)
    for key in seq_dict.keys():
        if key not in cluster_dict:
            raise Exception(f"Your metadata and a your fasta don't match,"
                            f" check your input\nCulprit {key}")
    total_seq = len(seq_dict.keys())
    return seq_dict, total_seq, cluster_dict, cluster_stats
