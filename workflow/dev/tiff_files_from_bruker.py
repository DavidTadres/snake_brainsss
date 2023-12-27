"""
Tiff files from Bruker come from RAW files which are ripped by a software from Bruker.
The tiff file has numbers going from 0 (no signal) to max 8191.
8191 is half of 2^14=16384. There seem to be no negative numbers in the tiff files.
"""