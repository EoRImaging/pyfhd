import numpy as np
from scipy.io import readsav
from pathlib import Path
import argparse


def splitter(sav_file, save_path):
    """
    The purpose of splitter is to take a large dictionary read from a sav file
    that contains many arrays as their values. These arrays will be imported
    as numpy arrays. Each key:value pair from the dictionary iterated through
    with the key acting as the file name, and the value is saved into the npy
    file. They will all be saved in a save_path, ideally this will be a data
    directory in the tests directory.

    This was made to prevent the reading of a large sav file for every test.
    This allows us to do the small tests quickly.

    Parameters
    ----------
    sav_file : Path
        The path to the save file i.e. /path/to/the/sav/file
    save_path : Path
        The path to the directory where we save all the .npy files i.e.
        /path/to/the/directory/where/we/want/save
        In the case the directory doesn't exist it will recursively make the
        directories to ensure it does.

    Raises
    ------
    OSError
        In the case the path for sav_file is not found and/or sav_file is not a file
        It will also get raised if the save_path is not a directory
    """

    # Ensure sav_file and save_path are of the right type, Path from pathlib
    sav_file = Path(sav_file)
    save_path = Path(save_path)
    # If the file is a file and exists
    if sav_file.exists() and sav_file.is_file():
        # Read the sav file
        dict_to_iter = readsav(sav_file, python_dict=True)
        # If the save_path doesn't exist create it (and its parents)
        if not save_path.exists():
            Path.mkdir(save_path, parents=True)
        # If save_path is a directory, then save the files, else raise an error
        if save_path.is_dir():
            # For every key, value pair, save a numpy file with key as the file
            # name in the save_path directory
            for key in dict_to_iter.keys():
                np.save(
                    str(save_path) + "/" + str(key) + ".npy",
                    dict_to_iter[key],
                    allow_pickle=True,
                )
                print(
                    str(key)
                    + ".npy"
                    + "\n\tHas been written to: "
                    + str(save_path)
                    + "/"
                    + str(key)
                    + ".npy"
                )
        else:
            raise OSError("save_path is not a dir")
    # If its not a file, then raise an OSError indicating that its not a file
    elif sav_file.exists() and not sav_file.is_file():
        raise OSError(
            "sav_file while it exists, is not a file, sav_file must be a file"
        )
    # If it doesn't exist then raise OSError indicating it doesn't exist
    else:
        raise OSError("sav_file does not exist")


if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sav_file", nargs="?", help="A sav file you want to split and save as npy files"
    )
    parser.add_argument(
        "save_path", nargs="?", help="Directory where all the .npy files will be saved"
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="In the case of multiple files use this argument instead",
    )
    parser.add_argument(
        "-d",
        "--directories",
        nargs="+",
        help="In the case you want to specifiy different directories for each "
        "file, use this argument.\nDirectories are taken in the same order as "
        "the files",
    )
    args = parser.parse_args()
    print(args)
    # If the sav_file is not None
    if args.sav_file is not None:
        # Check if the save_path is provided, if it is then use splitter
        if args.save_path is None:
            parser.error(
                "If you have provided a sav_file, you must also provide a save_path"
            )
        else:
            splitter(args.sav_file, args.save_path)
    # If multiple files were given then
    elif args.files is not None:
        # Check the directories argument
        if args.directories is None:
            # In the case its None, use save_path as many times as the size of files,
            # if save_path isn't provided, raise an error
            if args.save_path is None:
                parser.error("You must provide a path to save the files")
            else:
                directories = list(args.save_path) * len(args.files)
        if len(args.directories) != len(args.files):
            parser.error(
                "When using directories and files optional arguments, they must "
                "have the same number of options passed"
            )
        else:
            directories = args.directories
        # Combine files and directories together
        files_dirs = zip(args.files, directories, strict=True)
        # For each pair, call splitter
        for file, fdir in files_dirs:
            splitter(file, fdir)
    # No files given!
    else:
        parser.error("Please provide a file or files")
    print("\nAll files and directories written to successfully")
