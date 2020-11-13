from Bio import SeqIO
import os
data_set = '/Users/dolteanu/local_documents/MATLAB/DataBase/Primates/'

def preprocessing(data_set):
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
    # list of SeqIO dictionary like objects
    seqs = []
    cluster_names = os.listdir(data_set)
    # dictionary with Accession ID as keys and cluster name as values
    cluster_dict = {}
    # number of samples in each cluster
    cluster_samples_info = {}
    # count of the number of clusters as int
    number_of_clusters = len(cluster_names)
    type(number_of_clusters)
    # iterate through all the clusters' folders
    paths = []
    # Iterate over each cluster (top level directories)
    for cluster in cluster_names:
        accession_cluster_list = []
        files = os.listdir(os.path.join(data_set, cluster))
        files = [os.path.join(data_set, cluster, f) for f in files]
        paths.extend(files)
        # get path for cluster as str
        cluster_path = os.path.join(data_set, cluster)
        # get names of files in cluster as list of str
        file_name = os.listdir(cluster_path)
        # Iterate over each file in the cluster
        for file in file_name:
            # get path for each file in cluster as str
            file_path = os.path.join(cluster_path, file)
            # Required to use SeqIO.index to generate dictionaries to sort
            # accession ids by cluster, and get cluster sizes
            # SeqIO.index doesnt take file handle to index multiple seqs to a
            # single dict
            Seq_dict = SeqIO.index(file_path, "fasta")
            # Must iterate over keys to get dictionary to ensure
            for accession_id in Seq_dict.keys():
                cluster_dict.update({accession_id: cluster})
        # Not required, can get this from cluster_dict        accession_cluster_list.append(accession_id)
        # Dictionary for each cluster's sample count & list of accession ids
        # (for querying seqs)
        cluster_samples_info.update(
            {cluster: (len(accession_cluster_list), accession_cluster_list)})
    # Actual creation of sequences database (BioSQL) containing SeqRecords,
    # SeqIO.index_db doesnt allow the addition of info to the database,
    # otherwise we can bypass the use of SeqIO.index().
    seqs = SeqIO.index_db('Sequence_database.idx', paths, "fasta")
    print("%i sequences indexed in total" % len(seqs))
    # Add the number of files in cluster as dict of cluster name: file count
    total_seq = len(seqs)
    # cluster_dict no longer used can be removed
    return seqs, cluster_names, number_of_clusters, cluster_samples_info, total_seq, cluster_dict
