# MLDSP

Machine Learning with Digital Signal Processing (MLDSP) is an open-source, alignment-free, ultrafast, computationally
lightweight, and standalone software tool for comparison and analysis of DNA sequences. MLDSP is a general-purpose tool
that can be used for a variety of applications such as taxonomic classification, disease classification, virus subtype
classification, among others. It leverages the concept of a genomic signature extracted via a
Chaos Game Representation (CGR) of the DNA sequence. This is then used to generate an engineered feature vector for use
in supervised machine learning classification of samples. Currently sample classification is focused on model validation via 10-fold cross validation with sample prediction only in the context of an individual trained model, see below for details.

## Installing MLDSP command line interface

If you have access to the repository and you have an active python virtual environment, you can pip install MLDSP_core
using the following command on any unix based terminal shell:

```bash
pip install git+https://github.com/HillLab/MLDSP.git
```
This will give you access to the command-line interface.  
If you are having trouble installing it directly, clone the repo, checkout the dev_segio branch and pip install it:

```bash
git clone git@github.com:HillLab/MLDSP.git
cd MLDSP
pip install .
```

It should install all dependencies required for running

## Running MLDSP CLI

Once instlled simply call MLDSP:

```bash
$ MLDSP -h
usage: MLDSP train_set training_labels [options]

positional arguments:
  train_set             Path to dataset folder for training the ML models
  training_labels       CSV of metadata mapping class labels and sequence (fasta header) names

optional arguments:
  --help, -h            show this help message and exit
  --query_seq_path QUERY_DATA_DIRECTORY, -q QUERY_DATA_DIRECTORY
                        Path to query data folder for testing trained model prediction
  --run_name RUN_NAME, -r RUN_NAME
                        Unique name to track and store MLDSP run results (default: Bacteria)
  --output_directory OUTPUT_DIRECTORY, -o OUTPUT_DIRECTORY
                        Path to the output folder (default: current working directory)
  --cpus CPUS, -c CPUS  Number of cpu cores to use for parallel computation
                        (default: number of cores on your machine [int])
  --order ORDER, -d ORDER
                        Order of nucleotide vertex assignment in CGR, **DO NOT CHANGE** 
                        (default: ACGT)
  --kmer KMER, -k KMER  Kmer size for CGR (default: 7)
  --method [see __constants__.py string keys], -m [see __constants__.py string keys]
                        Numeric representation of method (see more in the
                        documentation) (default: 'Chaos Game Representation (CGR)')
  --folds FOLDS, -f FOLDS
                        Number of folds for cross-validation (default: 10)
  --dim_reduction ['pca', 'mds', 'tsne'], -i ['pca', 'mds', 'tsne']
                        Type of dimensionality reduction technique to use in the 
                        visualization of the training distance matrix. (default: mds)
  --quiet , -z          Option to supress command line output and redirect to prints.txt

```
`train_set` must be a folder containing .fasta files only to be used for MLDSP training (10X cross validation).  
Files can be single fasta or multi-fasta; header names are used to track samples, file names are discarded.  

`training_labels` must be a .csv file with NO HEADER and two columns; first column is sample names matching fasta headers from `train_set`, second column is supervised learning class labels.  

Both column values should be strings with no special characters: anything that is not an alphanumeric, dash, underscore, or dot will be removed; spaces and directory operators (/, \\) will be replaced with underscore. Original files will not be modified.  
`--query_seq_path` is the flag for providing unlabeled data for sample classification. Same conditions as `train_set` apply and class prediction i.e. sample classification can only be performed during a given not a posteriori.
## Notes
Additional sample predictions can be done from the Trained ML models but MLDSP CLI MUST be re run from the same location as previous with the same `--run name` and results output in the original output location (either default or same `--output_directory`). Optional arguments relating to CLI behaviour (`--quiet`,`--cpus`)