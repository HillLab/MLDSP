from Bio import SeqIO
import os

def preprocessing(dir):
    data = []
    directory = '/Users/dolteanu/local_documents/MATLAB/DataBase/Primates'
    cluster_names = os.listdir(directory)
    cluster_sample_count={}
    number_of_clusters = len(cluster_names)
    # if you know a more efficient way to get the files in a directory recursively please update Wanxin
    for cluster in cluster_names:
        data.append(cluster)
        cluster_path = os.path.join(directory, cluster)
        file_name = os.listdir(cluster_path)
        cluster_sample_count.update({cluster: len(file_name)})
        cluster_dict={}
        for file in file_name:
            file_path = os.path.join(cluster_path, file)
            cluster_dict.update(SeqIO.index(file_path, 'fasta'))

        data.append(cluster_dict)
    return data, cluster_names, number_of_clusters,
