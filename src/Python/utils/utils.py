import os
import glob
import subprocess

def get_gittop():
    """ This function returns the absolute path of current git repo root. """
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def remove_files_except(path, pattern):
    """ Removes all files in directory 'path' except some files that have the pattern 'pattern'. """
    all_list = glob.glob(os.path.join(path, "*"))
    remain_list = glob.glob(os.path.join(path, pattern))
    for f in list(set(all_list) - set(remain_list)):
        os.remove(f)

def get_saved_model_path_base():
    """ Returns the default project saved_model_path_base. (Global variable for multiple Python scripts.) """
    return  os.path.join(get_gittop(), "kernels", "templates")
