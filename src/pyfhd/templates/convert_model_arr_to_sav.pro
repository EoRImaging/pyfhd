PRO convert_model_arr_to_sav

    compile_opt idl2

    ;read in args
    args = COMMAND_LINE_ARGS()
    save_dir = args[0]
    obs_id = args[1]
    n_pol = FIX(args[2])

    ;this is the model visibilities as written out by pyfhd
    hdf5_filepath = save_dir + "/" + obs_id + "_vis_model.h5"

    ;load in the hdf5 file
    print, "Now loading model data from ", hdf5_filepath
    file_id = H5F_OPEN(hdf5_filepath)
    
    ;for as many polarisations as specified by n_pol, write out FHD style .sav
    ;files
    pol_names = ['XX', 'YY', 'XY', 'YX']

    ; for pol = 0, 0 do begin
    for pol = 0, n_pol-1 do begin

        ;read in this polarisation from the hdf5 file
        model_visi_data = H5D_OPEN(file_id, obs_id + "_vis_model_" + pol_names[pol])
        ;this reads into a struct containing real and imaginary as separate values
        model_visi_struct = H5D_READ(model_visi_data)

        ;things have to be saved inside a pointer array for FHD to load it back
        ;in correctly
        vis_model_ptr=PTRARR(1, /allocate)
        *vis_model_ptr[0] = COMPLEX(model_visi_struct.r, model_visi_struct.i)

        print, "Writing model uvfits to .sav file: " + obs_id + "_vis_model_" + pol_names[pol] + ".sav"

        ;save into the FHD vis_model format and naming convention
        idl_save = save_dir + "/" + obs_id + "_vis_model_" + pol_names[pol] + ".sav"
        save, vis_model_ptr, filename = idl_save

        ;close the hdf5 data struct, done with it now
        H5D_CLOSE, model_visi_data

    endfor
    ;close file, let's be tidy
    H5F_CLOSE, file_id

END
