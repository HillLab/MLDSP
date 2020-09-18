from Bio import SeqIO
import os
#directory = '/Users/dolteanu/local_documents/MATLAB/DataBase/Primates'

def preprocessing(directory):
    """Function to process fasta files in top-level directory, extract cluster
    info from dir names, and index (without ram loading) sequence data from fasta files into a
    BioPython SeqRecord class, this is in the nested list "data" with one internal list for each cluster with the 1st item being the cluster name & remaining items being the SeqRecord indices


    """
    data = []
    cluster_names = os.listdir(directory)
    cluster_sample_count={}
    # count of the number of clusters as int
    number_of_clusters = len(cluster_names)
    # iterate through all the clusters' folders
    # if you know a more efficient way to get the files in a directory recursively please update Wanxin
    for cluster in cluster_names:
        # add name of cluster as 1st item in list
        # get path for cluster as str
        cluster_samples = []
        cluster_samples.append(cluster)
        cluster_path = os.path.join(directory, cluster)
        # get names of files in cluster as list of str
        file_name = os.listdir(cluster_path)
        # Add the number of files in cluster as dict of cluster name: file count
        cluster_sample_count.update({cluster: len(file_name)})
        for file in file_name:
            # get path for each file in cluster as str
            file_path = os.path.join(cluster_path, file)
            # Add the fasta file's contents to a dict of
            # sequence accession : BioPython SeqRecord ()
            # without storing the dict in memory, this
            # is done in case we want to work with multi-
            # seq fastas in the future or other BioPython
            # data types
            Seq_record = SeqIO.index(file_path, "fasta")
            cluster_samples.append(Seq_record)
        data.append(cluster_samples)
    return data, cluster_names, number_of_clusters, cluster_sample_count
