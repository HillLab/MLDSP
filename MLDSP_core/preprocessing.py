from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple
import csv
from pyfaidx import Fasta
import re
import chardet


def csv2dict(infile: Path) -> Dict[str, str]:
    """
    Simple function to read the a csv into a dictionary (faster than pandas)
    Args:
        infile: Path to input file, comma delimited

    Returns:dictionary with first field as key and second as value
    """
    dictionary = {}
    # will not work if csv file is saved as utf-8 in excel
    with open(infile,newline='') as inf:
        dialect = csv.Sniffer().sniff(inf.read(2048))
        inf.seek(0)
        reader = csv.reader(inf,dialect)
        # assumes 1st column is accession and 2nd is class label
        for line in reader:
            key = line[0].replace("/","_").replace("\\","_")
            value = line[1].replace("/","_").replace("\\","_")
            dictionary[key] = value
    return dictionary
    

def preprocessing(data_set: Path, metadata: Optional[Path],
                  prefix: str = 'Train') -> Tuple[
    Fasta, int, Optional[Dict[str, str]], Optional[Counter]]:
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
    outfn = data_set.joinpath(f'{prefix}_all_seqs.fasta').resolve()
    if metadata is not None:
        cluster_dict = csv2dict(metadata)
        cluster_stats = Counter(cluster_dict.values())
    else:
        cluster_dict = cluster_stats = None
    if outfn.exists():
        print(f'File {outfn.name} exists and will be used! If this is '
              f'unintended, please remove the file\n')
    else:
        for file in data_set.glob('*'):
            if file.suffix != '.fai':
                with open(file) as infile, open(outfn, 'a') as outfile:
                    outfile.write(f'{infile.read().strip()}\n')
    seq_dict = Fasta(str(outfn),key_function = lambda x:x.replace("/","_").replace("\\","_"),sequence_always_upper=True)
    if metadata is not None:
        difference = set(cluster_dict.keys()).difference(seq_dict.keys())
        if difference:
            raise Exception(f"Your metadata and your fasta don't match,"
                            f" check your input\nCulprit(s): "
                            f"{''.join(difference)}")
    total_seq = len(seq_dict.keys())
    return seq_dict, total_seq, cluster_dict, cluster_stats
