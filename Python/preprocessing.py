from Bio import SeqIO
import os
directory = '/Users/dolteanu/local_documents/MATLAB/DataBase/Primates'

def preprocessing(directory):
    """Function to process fasta files in top-level directory, extract cluster
    info from dir names, and index (without ram loading) sequence data from
    fasta files. Output is 1. 'seqs' a list of SeqIO dictionary like objects
    each of which gives 1 SeqRecord object per sequence from that fasta file
    containing the seq string & associated info. 2. 'cluster_dict' a
    dictionary of accession ids as keys & cluster names as values
    """
    # list of SeqIO dictionary like objects
    seqs = []
    cluster_names = os.listdir(directory)
    # dictionary with Accession ID as keys and cluster name as values
    cluster_dict = {}
    # number of samples in each cluster
    cluster_sample_count = {}
    # count of the number of clusters as int
    number_of_clusters = len(cluster_names)
    # iterate through all the clusters' folders
    paths = []
    for cluster in cluster_names:
        files = os.listdir(os.path.join(directory, cluster))
        files = [os.path.join(directory, cluster, f) for f in files]
        paths.extend(files)
        # get path for cluster as str
        cluster_path = os.path.join(directory, cluster)
        # get names of files in cluster as list of str
        file_name = os.listdir(cluster_path)
        for file in file_name:
            # get path for each file in cluster as str
            file_path = os.path.join(cluster_path, file)
            # Add the fasta file's contents to a dict of
            # sequence accession : BioPython SeqRecord ()
            # without storing the dict in memory, this
            # is done in case we want to work with multi-
            # seq fastas in the future or other BioPython
            # data types
            Seq_dict = SeqIO.index(file_path, "fasta")
            # dictionary to identify which sequences belong to which clusters
            for accession_id in Seq_dict.keys():
                cluster_dict.update({accession_id: cluster})
        cluster_sample_count.update({cluster: len(cluster_dict)})
    seqs = SeqIO.index_db('Sequence_database.idx', paths, "fasta")
    # Add the number of files in cluster as dict of cluster name: file count
    total_seq = len(cluster_dict)
    return seqs, cluster_names, number_of_clusters, cluster_sample_count, total_seq, cluster_dict
