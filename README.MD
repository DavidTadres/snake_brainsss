# Install mamba
https://snakemake.readthedocs.io/en/stable/tutorial/setup.html

 conda activate base
 mamba create -c conda-forge -c bioconda -n snakemake snakemake

To activate env:
 conda activate snakemake

# Better
https://github.com/pimentel/sherlock-notes/blob/master/README.md

ml python/3.9.0 (on sherlock only)
make an environemnt and then do pip3 install snakemake

python3 -m venv .env_snakemake # Local Mac is 'snakemake'
source .env_snakemake/bin/activate
pip install snakemake
pip install nibabel
pip install natsort

to use:
ml python/3.9.0
source .env_snakemake/bin/activate