import os
import random
from Bio import SeqIO


def preprocessing(data_set, max_clust_size):
    """Preprocessing of fasta sequences using BioPython into a database of
    SeqRecord objects each representing a unique sequence, can handle multiple
    sequence fastas.

    seqs: main sequence database, dictionary-like object, no info on clusters
    cluster_names: list of all cluster names from sub-directory names.

    number_of_clusters: integer of the total count of clusters.

    cluster_sample_info: Dictionary with keys=cluster_names and values =
    a tuple consisting of: (number of samples in cluster,
        a list of accession ids corresponding to sequences of that cluster).

    total_seq: integer of the total sequence count in dataset.

    cluster_dict: depracated with cluster_sample_info, will be removed
    """
    # Dictionary to store SeqIO
    seq_dict = {}
    # seq_dict = h5py.File('seq_dict.hdf5', 'w')
    cluster_names = sorted(os.listdir(data_set))
    # dictionary with Accession ID as keys and cluster name as values
    cluster_dict = {}
    # number of samples in each cluster
    # cluster_samples_info = {}
    # count of the number of clusters as int
    cluster_stats={}
    # iterate through all the clusters' folders
    paths = []
    # Iterate over each cluster (top level directories)
    for cluster in cluster_names:
        files = os.listdir(os.path.join(data_set, cluster))
        files = [os.path.join(data_set, cluster, f) for f in files]
        paths.extend(files)
        # get path for cluster as str
        cluster_path = os.path.join(data_set, cluster)
        # get names of files in cluster as list of str
        file_name = sorted(os.listdir(cluster_path))
        cluster_stats.update({cluster:len(file_name)})
        temp_dict={}
        # Iterate over each file in the cluster
        for file in file_name:
            # get path for each file in cluster as str
            file_path = os.path.join(cluster_path, file)
            # Required to use SeqIO.index to generate dictionary of SeqRecords (parsed on demand in main script)

            # SeqIO.index doesnt take file handle to index multiple seqs to a
            # single dict

            # Not sure if storing the dict like object of SeqIO.index in a dict forces loading into memory (performance penalty)
            seqs = SeqIO.index(file_path, "fasta"
            #,key_function=get_accession
            )
            seq_dict.update(seqs)
            # Generate second dictionary for cluster info
            for accession_id in seqs.keys():
                cluster_dict.update({accession_id: cluster})
        # if len(temp_dict) >= max_clust_size:
        #     subset = dict(random.sample(temp_dict.items(), max_clust_size))
        #     print(len(subset))
        #     seq_dict.update(subset)
        # else:
        #     seq_dict.update(temp_dict)
        # for accession_id in seq_dict.keys():
        #     cluster_dict.update({accession_id: cluster})
    total_seq = len(seq_dict)
    del temp_dict 
    del seqs
    return seq_dict, cluster_stats, total_seq, cluster_dict

# accession_cluster_list.append(accession_id) # Not required, can get this from cluster_dict

# Not used because it generates a SQLite3 database which cannot be used with multiprocessing
# # Actual creation of sequences database (BioSQL) containing SeqRecords,
# # SeqIO.index_db doesnt allow the addition of info to the database,
# # otherwise we can bypass the use of SeqIO.index().
# seqs = SeqIO.index_db('Sequence_database.idx', paths, "fasta")
