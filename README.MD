# Installation on Sherlock
https://github.com/pimentel/sherlock-notes/blob/master/README.md (maybe useful info)

`ml python/3.9.0` (on sherlock only)

### make an environment

`python3 -m venv .env_snakemake` # Create environment (#Note for myself: Local Mac is '.env_snake_brainsss')

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

In your favorite folder on sherlock type:

`git clone https://github.com/DavidTadres/snake_brainsss` 

# Usage

## Building flies

Before the code works as expected, a config file with you name needs to be prepared. It can be
arbitrarily named, just remember the name.

The config file is in the folder 'users' and contains two lines:

1) The first line points to the fictrac data path.

2) The second line defines the root path where all the preprocessed data lives. 

See an existing file, i.e. 'dtadres.json' for an example.

Before building your first fly, please read this part carefully to make sure it'll run as expected:

1) fictrac: Your fictrac data must be in the folder define in the 'user.json' file. It must have the 
   following name: DATE_FLY_FUNC
   
   For example if you want to build the flies in the imports folder 
   "/oak/stanford/groups/trc/data/David/Bruker/imports/20240326" you'll see that there are 4 flies in 
   that folder.

   Each of those should have a fictrac folder. For the script to associate the correct fictrac data with
   the imaging data you must rename the fictrac folders to:
   "/oak/stanford/groups/trc/data/David/Bruker/Fictrac/20240328_fly1_func0"
   
   Since it's possible to have more than one 'func' folder per fly, you must define which imaging data 
   a particular fictrac dataset is associated with. 

2) Each fly must have a 'fly.json' in the 'imports' folder. For example, in  
   ""/oak/stanford/groups/trc/data/David/Bruker/imports/20240326/fly1" you will find such a 
   fly.json file. 

   The file contains 3 essential and several self-explanatory optional fields. Here we focus on 
   the essentials:

   a) genotype - this is essential because the flybuilder will group flies per genotype. For example,
   if genotype is "SS84990_DNa03_x_GCaMP6f" the fly builder will go to the 'dataset_folder' defined in the
   user.json (for dtadres.json that would be "/oak/stanford/groups/trc/data/David/Bruker/preprocessed") and
   create a folder "SS84990_DNa03_x_GCaMP6f" and deposit 'fly_001, fly_002...' in there.
   
   ! Hence, even a slight change in the genotype can be confusing as you'll create a lot of genotype folder.
   Make sure to be consistent !
   
   b) anatomy_channel - This is the static imaging channel: For example if we have tdTomato on Channel 1 and GCaMP 
   on channel 2, you would type "channel_1" (exactly like this!). There currently can only be a single 
   anatomy channel for snakebrainsss to work as expected. 
   
   c) functional_channel - This is the dynamic imaging channel. For example if we have tdTomato on Channel 1 and GCaMP 
   on channel 2, you would type "['channel_2']" (exactly like this!). 
   
   If you have two imaging channels, i.e.
   rGECI on Channel 1, gCarvi on channel 2 and mCardinal on Channel 3 you would type
   "['channel_1', 'channel_2']" (exactly like this!). Note that this hasn't been tested yet.

Now you are ready to build a fly. 
 
### Example: 

to build flies in this folder'/oak/stanford/groups/trc/data/David/Bruker/imports/20240326' 

On Sherlock type:

`ml python/3.9.0`

`source .env_snakemake/bin/activate`

cd to your snakebrainsss folder. If you don't know what this means try:

`cd snake_brainsss/workflow/`

Then type:
`snakemake --config user=dtadres --profile profiles/simple_slurm -s build_fly.smk --directory /oak/stanf
ord/groups/trc/data/David/Bruker/imports/20240326 -np`

The '-np' at the end makes this a test run. Check the output.

It will says which flies it wants to create in the genotype folder. If it looks ok, re-run the code
without the '-np' flag:  

`snakemake --config user=dtadres --profile profiles/simple_slurm -s build_fly.smk --directory /oak/stanf
ord/groups/trc/data/David/Bruker/imports/20240326`

Building the flies will take a few minutes (~2 minutes per fly)

You will get an email when the job starts and another email when it ends (or fails, cancels etc.) 

## Preprocess flies

Before continuing, make sure that the fictrac and imaging data is in the expected folder!

### WARNING ###
Visual stimulus not implemented TBD!
### WARNING ###
Fictrac fps is currently hardcoded to 100 fps (preprocess_fly.smk, line 37). To be changed in
the future
### WARNING ###

Once the fly is built, the preprocessing is straightforward. For example, to preprocess fl
"oak/stanford/groups/trc/data/David/Bruker/preprocessed/SS84990_DNa03_x_GCaMP6f/fly_037" you 
just type:

`snakemake --config user=dtadres --profile profiles/simple_slurm -s preprocess_fly.smk --directory /oak/stanf
ord/groups/trc/data/David/Bruker/preprocessed/SS84990_DNa03_x_GCaMP6f/fly_037`

This should take ~2 hours and you should have the same output as with the original brainsss.

You will get an email when the job starts and another email when it ends (or fails, cancels etc.). This is 
useful for debugging so that it's set to send email. To be discussed to turn it off in the future.

It's easy to automatically move all incoming emails into a given folder so that snakebrainsss doesn't
clog you inbox. 





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


# FAQ 
# On sherlock had warning message:

did:
pip list showed that I had urllib3==2.1.0 installed

https://stackoverflow.com/questions/76187256/importerror-urllib3-v2-0-only-supports-openssl-1-1-1-currently-the-ssl-modu
suggested to just do:

```shell
pip install urllib3==1.26.6
```
Warning indeed went away


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