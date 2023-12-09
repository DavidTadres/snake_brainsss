# Install mamba
https://snakemake.readthedocs.io/en/stable/tutorial/setup.html

 conda activate base
 mamba create -c conda-forge -c bioconda -n snakemake snakemake

To activate env:
 conda activate snakemake

# Better
https://github.com/pimentel/sherlock-notes/blob/master/README.md

(sherlock) module load python/3.XXXX
make an environemnt and then do pip3 install snakemake

python3 -m venv .snakemake
source .snakemake/bin/activate
pip install snakemake
pip install nibabel
pip install natsort