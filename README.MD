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

python3 -m venv .env_snakemake # Local Mac is '.env_snake_brainsss'
source .env_snakemake/bin/activate
pip install --upgrade pip
pip3 install antspyx
pip install snakemake
pip install natsort

pip install lxml
pip install openpyxl
pip install h5py
pip install pyfiglet
pip3 install opencv-python-headless
pip install scipy

# On sherlock had warning message:

did:
pip list showed that I had urllib3==2.1.0 installed
https://stackoverflow.com/questions/76187256/importerror-urllib3-v2-0-only-supports-openssl-1-1-1-currently-the-ssl-modu
suggested to just do:
```shell
pip install urllib3==1.26.6
```
Warning indeed went away

#pip install nibabel
cd .#pip install scipy
#pip install pandas
#
 

to use:
ml python/3.9.0
source .env_snakemake/bin/activate


To install on my M2 Mac (also see: https://github.com/ANTsX/ANTsPy/issues/519)
install homebrew: https://brew.sh/
```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install libpng
```
in environment:
pip install cmake
pip install itk
pip install Downloads/antspyx-0.3.8-cp310-cp310-macosx_13_0_arm64.whl
pip install snakemake
pip install h5py
pip install pyfiglet
pip install natsort

Create environment
```shell
python3 -m venv .env_snake_brainsss
source .env_snake_brainsss/bin/activate
```

download wheel for arm Mac https://github.com/dipterix/rpyANTs/releases/tag/0.0.1.9000
```shell
pip install YOUR WHEEL, for example: pip install antspyx-0.3.8-cp310-cp310-macosx_13_0_arm64.whl
```
# OR install python 3.9 and do:
python3 -m venv .env_snake_brainsss
source .env_snake_brainsss/bin/activate