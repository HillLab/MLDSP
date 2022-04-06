from codecs import EncodedFile
from collections import Counter
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from chardet import detect
from pyfaidx import Fasta


def csv2dict(infile: Path) -> Dict[str, str]:
    """
    Simple function to read the a csv into a dictionary (faster than pandas)
    Args:
        infile: Path to input file, comma delimited

    Returns:dictionary with first field as key and second as value
    """
    dictionary = {}
    # will not work if csv file is saved as utf-8 in excel in a MAC
    with open(infile, mode='rb') as reader:
        data = BytesIO(reader.read())
        data2 = deepcopy(data)
        en = detect(data.read())['encoding']
        reader = EncodedFile(data2, en, file_encoding='ascii')
        for line in reader:
            if line:
                line = line.strip().split(b',')
                key = line[0].replace(b"/", b"_").replace(b"\\", b"_").decode('ascii')
                value = line[1].replace(b"/", b"_").replace(b"\\", b"_").decode('ascii')
                dictionary[key] = value
    return dictionary


def preprocessing(data_set: Union[Path, str], metadata: Optional[Path],
                  prefix: str = 'Train') -> Tuple[Fasta, int,
                                                  Optional[Dict[str, str]],
                                                  Optional[Counter]]:
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
    def replace(x):
        return x.replace("/", "_").replace("\\", "_")

    data_set = Path(data_set).resolve()
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
            if file.suffix != '.fai' and '_all_seqs.fasta' not in str(file):
                with open(file) as infile, open(outfn, 'a') as outfile:
                    outfile.write(f'{infile.read().strip()}\n')
    seq_dict = Fasta(
        str(outfn), key_function=replace, duplicate_action="first",
        sequence_always_upper=True
    )
    if metadata is not None:
        difference = set(cluster_dict.keys()).difference(seq_dict.keys())
        if difference:
            raise Exception(f"Your metadata and your fasta don't match,"
                            f" check your input\nCulprit(s): "
                            f"{''.join(difference)}")
    total_seq = len(seq_dict.keys())
    return seq_dict, total_seq, cluster_dict, cluster_stats
