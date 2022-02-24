# MLDSP_core

Machine Learning with Digital Signal Processing is an open-source, alignment-free, ultrafast, computationally
lightweight, and standalone software tool for comparison and analysis of DNA sequences. MLDSP is a general-purpose tool
that can be used for a variety of applications such as taxonomic classification, disease classification, virus subtype
classification, evolutionary analyses, among others.

## Installing MLDSP command line

If you have access to the repository and you have an active python virtual environment, you can pip install MLDSP_core
on the dev_sergio branch by:

```bash
$ pip install git+https://github.com/HillLab/MLDSP.git@dev_sergio
```

If you are having trouble installing it directly, clone the repo, checkout the dev_segio branch and pip install it:

```bash
$ git clone git@github.com:HillLab/MLDSP.git
$ cd MLDSP
$ git switch dev_sergio
$ pip install .
```

It should install all dependencies required for running

## Running MLDSP_core

Once instlled simply call MLDSP:

```bash
$ MLDSP -h
usage: MLDSP_core data_set_path metadata [options]

positional arguments:
  training_set          Path to data set to train the models with
  training_labels       CSV with the mapping of labels and sequence names

optional arguments:
  -h, --help            show this help message and exit
  --run_name RUN_NAME, -r RUN_NAME
                        Name of the run (default: Bacteria)
  --output_directory OUTPUT_DIRECTORY, -o OUTPUT_DIRECTORY
                        Path to the output directory (default: .)
  --cpus CPUS, -c CPUS  Number of cpus to use (default: 12)
  --order ORDER, -d ORDER
                        Order of the nucleotides in CGR (default: ACGT)
  --kmer KMER, -k KMER  Kmer size (default: 5)
  --method {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, -m {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
                        Numeric representation of method (see more in the
                        documentation) (default: 14)
  --folds FOLDS, -f FOLDS
                        Number of folds for cross-validation (default: 10)
  --to_json, -j         Number of folds for cross-validation (default: False)

```