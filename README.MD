# Installation on Sherlock
https://github.com/pimentel/sherlock-notes/blob/master/README.md (maybe useful info)

`ml python/3.9.0` (on sherlock only)

### make an environment

`python3 -m venv .env_snakemake` # Create environment (# Local Mac is '.env_snake_brainsss')

`source .env_snakemake/bin/activate` # Activate environment

### Install necessary packages

`pip install --upgrade pip`

`pip3 install antspyx` # Install this first as it's the most fragile package

`pip install snakemake`

`pip install natsort`

`pip install lxml`

`pip install openpyxl`

`pip install h5py`

`pip install pyfiglet`

`pip3 install opencv-python-headless`

`pip install scipy`

It's currently best to make your own fork of my repository as there are elements that 
need to be changed from run to run in the source code. The goal is to not have to do that
and just be able to call snake_brains with the relevant arguments. TBD! 

In your favorite folder on sherlock type:

`git clone https://github.com/DavidTadres/snake_brainsss` # Or better, your own fork! 

# On sherlock had warning message:

did:
pip list showed that I had urllib3==2.1.0 installed

https://stackoverflow.com/questions/76187256/importerror-urllib3-v2-0-only-supports-openssl-1-1-1-currently-the-ssl-modu
suggested to just do:

```shell
pip install urllib3==1.26.6
```
Warning indeed went away

# Usage

## On your own PC: 

open terminal and type:

`git clone https://github.com/DavidTadres/snake_brainsss` # Or better, your own fork!

Open the repository on your laptop with some IDE, for example pycharm: https://www.jetbrains.com/pycharm/

In the folder 'snake_brains/workflow/users/' make sure there's a file with your sunet id, i.e. 'dtadres.json'

It needs to  have an imports_path and an dataset_path. Check dtadres.json for example

To build a fly, i.e. in '/oak/stanford/groups/trc/data/David/Bruker/imports/20240206__queue__' open the
file 'build_fly.smk'

Make sure that in line #19 you have `imports_to_process = ['20240206__queue__',]`

Make sure that in line #27 you have your sunet id, i.e. `current_user = 'dtadres'`

### WARNING ###
fictrac data currently problematic! TBD during meeting how to best deal with that data!

### WARNING ###
Visual stimulus not implemented TBD!

commit and push from your PC to gitlab.

## ON SHERLOCK

Prepare the environment and pull the changes from your PC

`ml python/3.9.0`

`source .env_snakemake/bin/activate`

`cd snake_brainsss/workflow`

`git pull`

To build the fly

`snakemake --profile profiles/sherlock -s build_fly.smk`

You will get an email when the job starts and another email when it ends (or fails, cancels etc.) 

After building the fly, check if all data is in the expected folders (fictrac, anat & func)

## On your own PC

To do preprocessing of the fly you just built, open the 'preprocess_fly.smk' file

Check line #31 and select the correct folder to process, for example `fly_folder_to_process = 'SS84990_DNa03_x_GCaMP6f/fly_005'`

Check line #37 for username: `current_user = 'dtadres'`

Check line 40 for fictrac fps: `fictrac_fps = 100`

git commit and push to github

## On Sherlock

If you just built a fly ignore the first >3< lines that follow:

`ml python/3.9.0`

`source .env_snakemake/bin/activate`

`cd snake_brainsss/workflow`

`git pull` < Do that in any case!

To run preprocessing

`snakemake --profile profiles/sherlock -s preprocess_fly.smk`

### Warning ### 
This should take ~1-2 hours. Do not lose the connection to sherlock (e.g. by shutting down 
the computer) or sherlock won't run the complete pipeline. Not a big problem as you can just restart
and everything that was done won't be run again, though.

# Other installation issues

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

# I don't think this is necessary!
(### Install mamba
https://snakemake.readthedocs.io/en/stable/tutorial/setup.html

 conda activate base
 mamba create -c conda-forge -c bioconda -n snakemake snakemake

To activate env:
 conda activate snakemake)

###
# Things that are very different to Bella's brainsss
###
1) the imaging files in 'func' and 'anat' folders all are called the same: instead of calling them 'functional_channel_1.nii' 
   or 'anatomical_channel_1.nii' everything is just called 'channel_1.nii'. Reason: Makes working with wildcards in snakemake
   much easier.
2) fly_builder enforces having a fly.json file. It MUST contain the following fields: 'Genotype', 'anatomy channel' and 
   'functional channel':
   - Genotype is used to create subfolders for each genotype which contain 'fly_001', 'fly_002' etc.
   - anatomy channel is a string with one of the following three values: 'channel1', 'channel2' or 'channel3'
   - functional channel is a list of strings, for example ['channel2',''], or ['channel1','channel2']
   - A given 'fly' will be assumed to always have the same anatomical and functional channels! I.e. if a fly is GCaMP and
     tdTomato, all recordings (of that fly, so that's anat, func0, func1 etc.) are assumed to be made with both the red
     and green channel!
3) correlation is done with a vectorized function I wrote instead of the scipy-pearson function which only works in 1D.
   Correlation now takes <1 minute. Result is expected to be almost identical (float32 as opposed to scipy float64) and
   precision can be increased if need be. 

###
# Small but important differences to Bella's brainsss
###
1) make_mean_brain was previously calculated with standard settings which casts a float64 array.
   For example: meanbrain = np.mean(brain_data, axis=-1), meanbrain.dtype -> dtype('float64')
   However, we never seem to be in float64 space so I changed it to:
   meanbrain = np.mean(brain_data, axis=-1, dtype=np.float32), meanbrain.dtype -> dtype('float32')

####
# Current limitations:
####
1) We assume that a given fly was imaged with a given set of channels. i.e. if fly001 has GCaMP and tdTomato it is assumed
   that ALL folders inside fly001 have data with two channels! 

###
# Questions:
###
1) I'm unsure about whether the conversion from uint16 to float32 is necessary and whether it's ok to convert back to 
   uint16 (happens for example when writing the h5 file back to a nii file).  
2) Why use nibabel at all? e.g. motion correction ants.registration takes a numpy array it seems. > Probably because
   it can be opened natively in fiji?
3) align_anat calls: Why do we provide a resolution (e.g. res_ant = (0.653, 0.653, 1)) and then resample with (2,2,2)? 
   It helps with plotting! Bella made the FDA 2um isotropic. The resolution would be higher but we don't really see more
   and it make plotting easier/faster

###
# Other notes:
###
1) Dangerous: in fictrac_utils.smooth_and_interp_fictrac the scipy.signal.savgol filter has a fixed smoothing
   length that does NOT depend on the framerate. This will yield different smoothing results depending on the 
   recording framerate!
2) Check line 146/147 in motion_correction.py in brainsss. Data is loaded as uint16 and then converted to float32
   Meanbrain is made by calling np.mean which returns a float64 array. I think this might lead to a loss of precision
   Better to avoid using uint16 in line 146 and use directly float32.  
3) Dangerous: line 61 in 'temporal_high_pass_filter.py' in brainsss - sigma is defined as 200 and not dependent on the speed
   at which data was collected! I think this will yield different filtering results depending on the aquisition speed!
4) Unclear what's happening/what the consequence is: in align_anat.py starting line 135: the variable is called 
   fwdtranforms_save_dirs but it contains moco['invtransforms']. I think it does copy invtransforms but not 100% sure.

###
# Todo:
###
1) Try parallelization of motion correction. This is by far the slowest step and would profit greatly from 
   parallelization. -> Done
2) Test what happens if we make supervoxels per volume (instead of z-slice). TBD but reason for supervoxel
   per z-slice is to keep temporal resolution as high as possible: Each slice is recorded ~20 ms later compared
   to the previous one. If we do 3D supervoxels we'll time-smear 
3) Keep track of dtypes more consistenly. -> Done
4) Correct axis metadata in nii files! Problem is that for us z=0 is posterior whereas for field it's anterior.
5) Resultion/voxel size should go into the nii file which should help with alignments.