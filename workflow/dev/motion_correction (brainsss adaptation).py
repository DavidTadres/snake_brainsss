# Cleaning up the preprocessing.py file
# After introducing the parallel processing moco don't need to keep this
# serial function in that module
def motion_correction(
    fly_directory,
    dataset_path,
    meanbrain_path,
    type_of_transform,
    output_format,
    flow_sigma,
    total_sigma,
    aff_metric,
    h5_path,
    anatomy_channel,
    functional_channels,
):
    """
    Motion correction function, essentially a copy from Bella's motion_correction.py script.
    The 'anatomy' and 'functional' channel(s) are defined in the 'fly.json' file of a given experimental folder.
    Here, we assume that there is a single 'anatomy' channel per experiment but it's possible to have zero, one or two
    functional channels.

    TODO: - ants seems to have a motion correction function. Try it.

    :param dataset_path: A list of paths
    :return:
    :param fly_directory:
    :param dataset_path:
    :param meanbrain_path:
    :param type_of_transform:
    :param output_format:
    :param flow_sigma:
    :param total_sigma:
    :param aff_metric:
    :param h5_path:
    :param anatomy_channel:
    :param functional_channels:
    :return:
    """

    #####################
    ### SETUP LOGGING ###
    #####################

    logfile = utils.create_logfile(fly_directory, function_name="motion_correction")
    printlog = getattr(utils.Printlog(logfile=logfile), "print_to_log")
    #utils.print_function_start(logfile, "motion_correction")

    print("dataset_path" + repr(dataset_path))

    #####
    # CONVERT PATHS TO PATHLIB.PATH OBJECTS
    #####
    dataset_path = utils.convert_list_of_string_to_posix_path(dataset_path)
    meanbrain_path = utils.convert_list_of_string_to_posix_path(meanbrain_path)
    h5_path = utils.convert_list_of_string_to_posix_path(h5_path)

    parent_path = dataset_path[
        0
    ].parent  # the path witout filename, i.e. ../fly_001/func1/imaging/

    ####
    # Identify which channel is the anatomy, or 'master' channel
    ####

    # This first loop assigns a pathlib.Path object to the path_brain_anatomy variable
    # Here we assume that there is ONLY ONE anatomy channel
    path_brain_anatomy = None
    for current_brain_path, current_meanbrain_path in zip(dataset_path, meanbrain_path):
        if (
            "channel_1.nii" in current_brain_path.name
            and "channel_1" in anatomy_channel
        ):
            if path_brain_anatomy is None:
                path_brain_anatomy = current_brain_path
                path_meanbrain_anatomy = current_meanbrain_path
            else:
                printlog(
                    "!!!! ANATOMY CHANNEL AREADY DEFINED AS "
                    + repr(path_brain_anatomy)
                    + "!!! MAKE SURE TO HAVE ONLY ONE ANATOMY CHANNEL IN FLY.JSON"
                )
        elif (
            "channel_2.nii" in current_brain_path.name
            and "channel_2" in anatomy_channel
        ):
            if path_brain_anatomy is None:
                path_brain_anatomy = current_brain_path
                path_meanbrain_anatomy = current_meanbrain_path
            else:
                printlog(
                    "!!!! ANATOMY CHANNEL AREADY DEFINED AS "
                    + repr(path_brain_anatomy)
                    + "!!! MAKE SURE TO HAVE ONLY ONE ANATOMY CHANNEL IN FLY.JSON"
                )
        elif (
            "channel_3.nii" in current_brain_path.name
            and "channel_3" in anatomy_channel
        ):
            if path_brain_anatomy is None:
                path_brain_anatomy = current_brain_path
                path_meanbrain_anatomy = current_meanbrain_path
            else:
                printlog(
                    "!!!! ANATOMY CHANNEL AREADY DEFINED AS "
                    + repr(path_brain_anatomy)
                    + "!!! MAKE SURE TO HAVE ONLY ONE ANATOMY CHANNEL IN FLY.JSON"
                )

    # Next, check if there are any functional channels.
    path_brain_functional = []
    path_meanbrain_functional = []
    for current_brain_path, current_meanbrain_path in zip(dataset_path, meanbrain_path):
        if (
            "channel_1.nii" in current_brain_path.name
            and "channel_1" in functional_channels
        ):
            path_brain_functional.append(current_brain_path)
            path_meanbrain_functional.append(current_meanbrain_path)
        if (
            "channel_2.nii" in current_brain_path.name
            and "channel_2" in functional_channels
        ):
            path_brain_functional.append(current_brain_path)
            path_meanbrain_functional.append(current_meanbrain_path)
        if (
            "channel_3.nii" in current_brain_path.name
            and "channel_3" in functional_channels
        ):
            path_brain_functional.append(current_brain_path)
            path_meanbrain_functional.append(current_meanbrain_path)

    ####
    # DEFINE SAVEPATH
    ####

    # This could be integrated in loop above but for readability I'll make extra loops here
    # First, path to resulting anatomy motion corrected file.
    for current_h5_path in h5_path:
        if "channel_1" in current_h5_path.name and "channel_1" in anatomy_channel:
            path_h5_anatomy = current_h5_path
        elif "channel_2" in current_h5_path.name and "channel_2" in anatomy_channel:
            path_h5_anatomy = current_h5_path
        elif "channel_3" in current_h5_path.name and "channel_3" in anatomy_channel:
            path_h5_anatomy = current_h5_path

    # functional path is optional - to empty list in case no functional channel is provided
    path_h5_functional = []
    for current_h5_path in h5_path:
        if "channel_1" in current_h5_path.name and "channel_1" in functional_channels:
            path_h5_functional.append(current_h5_path)
        if "channel_2" in current_h5_path.name and "channel_2" in functional_channels:
            path_h5_functional.append(current_h5_path)
        if "channel_3" in current_h5_path.name and "channel_3" in functional_channels:
            path_h5_functional.append(current_h5_path)

    # Print useful info to log file
    printlog(f"Dataset path{parent_path.name:.>{WIDTH - 12}}")
    printlog(f"Brain anatomy{path_brain_anatomy.name:.>{WIDTH - 12}}")
    if len(path_brain_functional) > 0:
        printlog(f"Brain functional{str(path_brain_functional[0].name):.>{WIDTH - 12}}")
        if len(path_brain_functional) > 1:
            printlog(
                f"Brain functional#2{str(path_brain_functional[1].name):.>{WIDTH - 12}}"
            )
    else:
        printlog("No functional path found {:.>{WIDTH - 12}}")

    printlog(f"type_of_transform{type_of_transform:.>{WIDTH - 17}}")
    printlog(f"output_format{output_format:.>{WIDTH - 13}}")
    printlog(f"flow_sigma{flow_sigma:.>{WIDTH - 10}}")
    printlog(f"total_sigma{total_sigma:.>{WIDTH - 11}}")


    # This can't really happen with snakemake as it won't even start the job without
    # the input files present!

    ########################################
    ### Read Channel 1 imaging data ###
    ########################################
    ### Get Brain Shape ###
    img_anatomy = nib.load(path_brain_anatomy)  # this loads a proxy
    anatomy_shape = img_anatomy.header.get_data_shape()
    brain_dims = anatomy_shape
    printlog(f"Master brain shape{str(brain_dims):.>{WIDTH - 18}}")

    ################################
    ### DEFINE STEPSIZE/CHUNKING ###
    ################################
    # Ideally, we define the number of slices on the (x,y) size of the images!

    path_brain_anatomy
    if "func" in path_brain_anatomy.parts:
        scantype = "func"
        stepsize = 100  # Bella had this at 100
    elif "anat" in path_brain_anatomy.parts:
        scantype = "anat"
        stepsize = 5  # Bella had this at 5
    else:
        scantype = "Unknown"
        stepsize = 100
        printlog(
            f"{'   Could not determine scantype. Using default stepsize of 100   ':*^{WIDTH}}"
        )
    printlog(f"Scantype{scantype:.>{WIDTH - 8}}")
    printlog(f"Stepsize{stepsize:.>{WIDTH - 8}}")


    ########################################
    ### Read Meanbrain of Channel 1 ###
    ########################################

    meanbrain_proxy = nib.load(path_meanbrain_anatomy)
    #meanbrain = np.asarray(meanbrain_proxy.dataobj, dtype=np.uint16)
    meanbrain = np.asarray(meanbrain_proxy.dataobj, dtype=DTYPE)
    # get_fdata() loads data into memory and sometimes doesn't release it.
    #fixed_ants = ants.from_numpy(np.asarray(meanbrain, dtype="float32"))
    fixed_ants = ants.from_numpy(np.asarray(meanbrain, dtype=DTYPE))
    printlog(f"Loaded meanbrain{path_meanbrain_anatomy.name:.>{WIDTH - 16}}")

    #########################
    ### Load Mirror Brain ###
    #########################

    if len(path_brain_functional) > 0:  # is not None:
        img_functional_one_proxy = nib.load(path_brain_functional[0])  # this loads a proxy
        # make sure channel anatomy and functional have same shape
        functional_one_shape = img_functional_one_proxy.header.get_data_shape()
        if anatomy_shape != functional_one_shape:
            printlog(
                f"{'   WARNING Channel anatomy and functional do not have the same shape!   ':*^{WIDTH}}"
            )
            printlog("{} and {}".format(anatomy_shape, functional_one_shape))

        # Once we have the 3 detector channel in the bruker, we'll have 2 functional channels
        if len(path_brain_functional) > 1:
            img_functional_two = nib.load(
                path_brain_functional[1]
            )  # this loads a proxy
            # make sure both functional channels have same dims
            functional_two_shape = img_functional_two.header.get_data_shape()
            if functional_one_shape != functional_two_shape:
                printlog(
                    f"{'   WARNING Channel functional one and functional two do not have the same shape!   ':*^{WIDTH}}"
                )
                printlog("{} and {}".format(functional_one_shape, functional_two_shape))
    ############################################################
    ### Make Empty MOCO files that will be filled vol by vol ###
    ############################################################
    MEMORY_ONLY = False # IF False, will chunk, if true will currently
    if not MEMORY_ONLY:
        # This should most likely live on scratch as it is accessed several times.
        # h5_file_name = f"{path_brain_master.name.split('.')[0]}_moco.h5"
        # moco_dir, savefile_master = brainsss.make_empty_h5(h5_path_scratch, h5_file_name, brain_dims)#, save_type)
        # Make 'moco' dir in imaging path
        path_h5_anatomy.parent.mkdir(parents=True, exist_ok=True)
        # Create empty h5 file
        with h5py.File(path_h5_anatomy, "w") as file:
            _ = file.create_dataset("data", brain_dims, dtype="float32", chunks=True)
        printlog(f"Created empty hdf5 file{path_h5_anatomy.name:.>{WIDTH - 23}}")

        if len(path_h5_functional) > 0:
            with h5py.File(path_h5_functional[0], "w") as file:
                _ = file.create_dataset("data", brain_dims, dtype="float32", chunks=True)
            printlog(f"Created empty hdf5 file{path_h5_functional[0].name:.>{WIDTH - 23}}")

            if len(path_h5_functional) > 1:
                with h5py.File(path_h5_functional[1], "w") as file:
                    _ = file.create_dataset(
                        "data", brain_dims, dtype="float32", chunks=True
                    )
                printlog(
                    f"Created empty hdf5 file{path_h5_functional[1].name:.>{WIDTH - 23}}"
                )

    #################################
    ### Perform Motion Correction ###
    #################################
    printlog(f"{'   STARTING MOCO   ':-^{WIDTH}}")
    transform_matrix = []

    ### prepare chunks to loop over ###
    # the chunks defines how many vols to moco before saving them to h5 (this save is slow, so we want to do it less often)
    steps = list(range(0, brain_dims[-1], stepsize))
    # add the last few volumes that are not divisible by stepsize
    if brain_dims[-1] > steps[-1]:
        steps.append(brain_dims[-1])

    # loop over all brain vols, motion correcting each and insert into hdf5 file on disk
    # for i in range(brain_dims[-1]):
    start_time = time.time()
    print_timer = time.time()

    # We should have enough RAM to keep everything in memory -
    # It seems to be slow to write to h5py unless it's optimized - not sure if that's one reason why moco is so slow:
    # https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage

    if MEMORY_ONLY:
        # Input be 4D array! Else fails here. This is preallocation of memory for the resulting motion-corrected
        # anatomy channel
        moco_anatomy = np.zeros((brain_dims[0], brain_dims[1], brain_dims[2], brain_dims[3]), dtype=np.float32)
        moco_anatomy.fill(np.nan) # avoid that missing values end up as 0!
        #
        transform_matrix = np.zeros((12, brain_dims[3])) # Lets see if that's correct

        if len(path_brain_functional) > 0:
            # This is preallocation of memory for the resulting motion-corrected functional channel
            moco_functional_one = np.zeros((brain_dims[0], brain_dims[1], brain_dims[2], brain_dims[3]), dtype=np.float32)
            moco_functional_one.fill(np.nan)

        # Register one frame after another
        for current_frame in range(brain_dims[-1]): # that's the t-dimension
            current_moving_frame = img_anatomy.dataobj[:,:,:,current_frame] # load a single frame into memory (memory-cheap, but i/o heavy)
            current_moving_frame_ants = ants.from_numpy(np.asarray(current_moving_frame, dtype=np.float32)) # cast np.uint16 as float32 because ants requires it

            moco = ants.registration(
                    fixed_ants,
                    current_moving_frame_ants,
                    type_of_transform=type_of_transform,
                    flow_sigma=flow_sigma,
                    total_sigma=total_sigma,
                    aff_metric=aff_metric,
                    #outputprefix='' # MAYBE writing on scratch will make this faster?
            )
            moco_anatomy[:,:,:,current_frame] = moco["warpedmovout"].numpy()
            # Get transform info, for saving and applying transform to functional channel
            transformlist = moco["fwdtransforms"]

            ### APPLY TRANSFORMS TO FUNCTIONAL CHANNELS ###
            if len(path_brain_functional) > 0:
                current_functional_frame = img_functional_one_proxy.dataobj[:,:,:, current_frame]
                current_functional_frame_ants = ants.from_numpy(
                    np.asarray(current_functional_frame, dtype=np.float32)
                )
                moco_functional = ants.apply_transforms(
                    fixed_ants, current_functional_frame_ants, transformlist
                )
                print("fixed_ants " + repr(fixed_ants))
                moco_functional_one[:,:,:,current_frame] = moco_functional.numpy()
                # TODO add more channels here!!!

            # Delete transform info - might be worth keeping instead of huge resulting file? TBD
            for x in transformlist:
                if ".mat" in x:
                    temp = ants.read_transform(x)
                    transform_matrix[:, current_frame] = temp.parameters
                # lets' delete all files created by ants - else we quickly create thousands of files!
                pathlib.Path(x).unlink()

            ### Print progress ###
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 * 60:  # if less than 1 min has elapsed
                print_frequency = (
                    1  # print every sec if possible, but will be every vol
                )
            elif elapsed_time < 5 * 60:
                print_frequency = 1 * 60
            elif elapsed_time < 30 * 60:
                print_frequency = 5 * 60
            else:
                print_frequency = 60 * 60
            if time.time() - print_timer > print_frequency:
                print_timer = time.time()
                moco_utils.print_progress_table_moco(
                    total_vol=brain_dims[-1],
                    complete_vol=current_frame,
                    printlog=printlog,
                    start_time=start_time,
                    width=WIDTH,
                )

        aff = np.eye(4)
        anatomy_save_object = nib.Nifti1Image(moco_anatomy, aff)
        anatomy_save_object.to_filename(path_h5_anatomy)
        if len(path_brain_functional) > 0:
            functional_one_save_object = nib.Nifti1Image(moco_functional_one)
            functional_one_save_object.to_filename(path_h5_functional[0])

    if not MEMORY_ONLY:
        # For timepoints / stepsize. e.g. if have 300 timepoints and stepsize 100 I get len(steps)=3
        for j in range(len(steps) - 1):
            # printlog(F"j: {j}")

            ### LOAD A SINGLE BRAIN VOL ###
            moco_anatomy_chunk = [] # This really should be a preallocated array!!!
            moco_functional_one_chunk = [] # This really should be a preallocated array!!!
            moco_functional_two_chunk = [] # This really should be a preallocated array!!!
            # for each timePOINT!
            for i in range(stepsize):
                # that's a number
                index = steps[j] + i
                # for the very last j, adding the step size will go over the dim, so need to stop here
                if index == brain_dims[-1]:
                    break
                # that's a single slice in time
                vol = img_anatomy.dataobj[..., index] #
                moving = ants.from_numpy(np.asarray(vol, dtype="float32"))

                ### MOTION CORRECT ###
                # This step doesn't seem to take very long - ~2 seconds on a 256,128,49 volume
                moco = ants.registration(
                    fixed_ants,
                    moving,
                    type_of_transform=type_of_transform,
                    flow_sigma=flow_sigma,
                    total_sigma=total_sigma,
                    aff_metric=aff_metric,
                )
                # moco_ch1 = moco['warpedmovout'].numpy()
                moco_anatomy = moco["warpedmovout"].numpy()
                # moco_ch1_chunk.append(moco_ch1)
                moco_anatomy_chunk.append(moco_anatomy) # This really should be a preallocated array!!!
                transformlist = moco["fwdtransforms"]
                # printlog(F'vol, ch1 moco: {index}, time: {time.time()-t0}')

                ### APPLY TRANSFORMS TO FUNCTIONAL CHANNELS ###
                if len(path_brain_functional) > 0:
                    vol = img_functional_one_proxy.dataobj[..., index]
                    functional_one_moving = ants.from_numpy(
                        np.asarray(vol, dtype="float32")
                    )
                    moco_functional_one = ants.apply_transforms(
                        fixed_ants, functional_one_moving, transformlist
                    )
                    moco_functional_one = moco_functional_one.numpy()
                    moco_functional_one_chunk.append(moco_functional_one)
                    # If a second functional channel exists, also apply to this one
                    if len(path_brain_functional) > 1:
                        vol = img_functional_two.dataobj[..., index]
                        functional_two_moving = ants.from_numpy(
                            np.asarray(vol, dtype="float32")
                        )
                        moco_functional_two = ants.apply_transforms(
                            fixed_ants, functional_two_moving, transformlist
                        )
                        moco_functional_two = moco_functional_two.numpy()
                        moco_functional_two_chunk.append(moco_functional_two)
                ### APPLY TRANSFORMS TO CHANNEL 2 ###
                # t0 = time.time()
                # if path_brain_mirror is not None:
                #    vol = img_ch2.dataobj[..., index]
                #    ch2_moving = ants.from_numpy(np.asarray(vol, dtype='float32'))
                #    moco_ch2 = ants.apply_transforms(fixed, ch2_moving, transformlist)
                #    moco_ch2 = moco_ch2.numpy()
                #    moco_ch2_chunk.append(moco_ch2)
                # printlog(F'moco vol done: {index}, time: {time.time()-t0}')

                ### SAVE AFFINE TRANSFORM PARAMETERS FOR PLOTTING MOTION ###
                transformlist = moco["fwdtransforms"]
                for x in transformlist:
                    if ".mat" in x:
                        temp = ants.read_transform(x)
                        transform_matrix.append(temp.parameters)

                ### DELETE FORWARD TRANSFORMS ###
                transformlist = moco["fwdtransforms"]
                for x in transformlist:
                    if ".mat" not in x:
                        # print('Deleting fwdtransforms ' + x) # Yes, these are files
                        pathlib.Path(x).unlink()
                        # os.remove(x) # todo Save memory? # I think that because otherwise we'll
                        # quickly have tons of files in the temp folder which will make it hard to
                        # know which one we are looking for.

                ### DELETE INVERSE TRANSFORMS ###
                transformlist = moco[
                    "invtransforms"
                ]  # I'm surprised this doesn't lead to an error because it doesn't seem taht moco['invtransforms'] is defined anywhere
                for x in transformlist:
                    if ".mat" not in x:
                        # print('Deleting invtransforms ' + x)
                        pathlib.Path(x).unlink()
                        # os.remove(x) # todo Save memory? # I think that because otherwise we'll
                        # quickly have tons of files in the temp folder which will make it hard to
                        # know which one we are looking for.
                        # No it seems mainly a memory/filenumber issue: for each registration call (so several
                        # 1000 times per call of this function) a >10Mb file is created. That adds of course

                ### Print progress ###
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 * 60:  # if less than 1 min has elapsed
                    print_frequency = (
                        1  # print every sec if possible, but will be every vol
                    )
                elif elapsed_time < 5 * 60:
                    print_frequency = 1 * 60
                elif elapsed_time < 30 * 60:
                    print_frequency = 5 * 60
                else:
                    print_frequency = 60 * 60
                if time.time() - print_timer > print_frequency:
                    print_timer = time.time()
                    moco_utils.print_progress_table_moco(
                        total_vol=brain_dims[-1],
                        complete_vol=index,
                        printlog=printlog,
                        start_time=start_time,
                        width=WIDTH,
                    )

            moco_anatomy_chunk = np.moveaxis(np.asarray(moco_anatomy_chunk), 0, -1)
            if len(path_brain_functional) > 0:
                moco_functional_one_chunk = np.moveaxis(
                    np.asarray(moco_functional_one_chunk), 0, -1
                )
                if len(path_brain_functional) > 1:
                    moco_functional_two_chunk = np.moveaxis(
                        np.asarray(moco_functional_two_chunk), 0, -1
                    )
            # moco_ch1_chunk = np.moveaxis(np.asarray(moco_ch1_chunk), 0, -1)
            # if path_brain_mirror is not None:
            #    moco_ch2_chunk = np.moveaxis(np.asarray(moco_ch2_chunk), 0, -1)
            # printlog("chunk shape: {}. Time: {}".format(moco_ch1_chunk.shape, time.time()-t0))

            ### APPEND WARPED VOL TO HD5F FILE - CHANNEL 1 ###
            with h5py.File(path_h5_anatomy, "a") as f:
                f["data"][..., steps[j] : steps[j + 1]] = moco_anatomy_chunk
            # t0 = time.time()
            # with h5py.File(path_h5_master, 'a') as f:
            #    f['data'][..., steps[j]:steps[j + 1]] = moco_ch1_chunk
            # printlog(F'Ch_1 append time: {time.time-t0}')

            ### APPEND WARPED VOL TO HD5F FILE - CHANNEL 2 ###
            if len(path_brain_functional) > 0:
                with h5py.File(path_h5_functional[0], "a") as f:
                    f["data"][..., steps[j] : steps[j + 1]] = moco_functional_one_chunk
                if len(path_brain_functional) > 1:
                    with h5py.File(path_h5_functional[1], "a") as f:
                        f["data"][..., steps[j] : steps[j + 1]] = moco_functional_two_chunk
            # t0 = time.time()
            # if path_brain_mirror is not None:
            #    with h5py.File(path_h5_mirror, 'a') as f:
            #        f['data'][..., steps[j]:steps[j + 1]] = moco_ch2_chunk
            # printlog(F'Ch_2 append time: {time.time()-t0}')

    ### SAVE TRANSFORMS ###
    printlog("saving transforms")
    printlog(f"path_h5_master: {path_h5_anatomy}")
    transform_matrix = np.array(transform_matrix)
    # save_file = os.path.join(moco_dir, 'motcorr_params')
    save_file = pathlib.Path(path_h5_anatomy.parent, "motcorr_params")
    np.save(save_file, transform_matrix)

    ### MAKE MOCO PLOT ###
    printlog("making moco plot")
    printlog(f"moco_dir: {path_h5_anatomy.parent}")
    moco_utils.save_moco_figure(
        transform_matrix=transform_matrix,
        parent_path=parent_path,
        moco_dir=path_h5_anatomy.parent,
        printlog=printlog,
    )

    ### OPTIONAL: SAVE REGISTERED IMAGES AS NII ###
    if output_format == "nii":
        printlog("saving .nii images")

        # Save master:
        nii_savefile_master = moco_utils.h5_to_nii(path_h5_anatomy)
        printlog(f"nii_savefile_master: {str(nii_savefile_master.name)}")
        if (
            nii_savefile_master is not None
        ):  # If .nii conversion went OK, delete h5 file
            printlog("deleting .h5 file at {}".format(path_h5_anatomy))
            path_h5_anatomy.unlink()  # delete file
        else:
            printlog("nii conversion failed for {}".format(path_h5_anatomy))
        # Save mirror:
        if len(path_h5_functional) > 0:
            nii_savefile_functional_one = moco_utils.h5_to_nii(path_h5_functional[0])
            printlog(f"nii_savefile_mirror: {str(nii_savefile_functional_one)}")
            if (
                nii_savefile_functional_one is not None
            ):  # If .nii conversion went OK, delete h5 file
                printlog("deleting .h5 file at {}".format(path_h5_functional[0]))
                # os.remove(savefile_mirror)
                path_h5_functional[0].unlink()
            else:
                printlog("nii conversion failed for {}".format(path_h5_functional[0]))

            if len(path_h5_functional) > 1:
                nii_savefile_functional_two = moco_utils.h5_to_nii(
                    path_h5_functional[1]
                )
                printlog(f"nii_savefile_mirror: {str(nii_savefile_functional_two)}")
                if (
                    nii_savefile_functional_two is not None
                ):  # If .nii conversion went OK, delete h5 file
                    printlog("deleting .h5 file at {}".format(path_h5_functional[1]))
                    # os.remove(savefile_mirror)
                    path_h5_functional[1].unlink()
                else:
                    printlog(
                        "nii conversion failed for {}".format(path_h5_functional[1])
                    )
        # Save mirror:
        # if path_brain_mirror is not None:
        #    nii_savefile_mirror = moco_utils.h5_to_nii(path_h5_functional[0])
        #    printlog(F"nii_savefile_mirror: {str(nii_savefile_mirror)}")
        #    if nii_savefile_mirror is not None:  # If .nii conversion went OK, delete h5 file
        #        printlog('deleting .h5 file at {}'.format(path_h5_mirror))
        #        #os.remove(savefile_mirror)
        #        path_h5_mirror.unlink()
        #    else:
        #        printlog('nii conversion failed for {}'.format(path_h5_mirror))

