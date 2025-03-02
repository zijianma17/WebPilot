
''''
this scripts runs the test on webarena, including some global settings and may call any other methods
'''

import json
import os
import re
import subprocess
import time
import logging
import pickle # for MCTS class saving
import shutil
import traceback
import sys


def remove_generated_files_id(task_id):
    """
    remove the generated files. [task_id]_*.gv, [task_id]_*.gv.pdf, root_[task_id].pkl
    """
    task_id = str(task_id)

    # delete all [task_id]_*.gv files in the current folder
    for file in os.listdir("."):
        if file.startswith(task_id) and file.endswith(".gv"):
            os.remove(file)

    # delete all [task_id]_*.gv.pdf files in the current folder
    for file in os.listdir("."):
        if file.startswith(task_id) and file.endswith(".gv.pdf"):
            os.remove(file)

    # delete [task_id]_root.pkl file in the current folder
    root_file = f"root_{task_id}.pkl"
    if os.path.exists(root_file):
        os.remove(root_file)

def remove_generated_files():
    """
    remove the generated files, including .gv and .gv.pdf, .png, single_cycle.log, root.pkl
    """
    # delete the files only in the current folder
    for file in os.listdir("."):
        if file.endswith(".gv") or file.endswith(".gv.pdf") or file.endswith(".png")\
            or file.endswith(".pkl"):
            os.remove(file)
    
    if os.path.exists("root.pkl"):
        os.remove("root.pkl")

    # delete all root_*.pkl files in the current folder
    for file in os.listdir("."):
        if file.startswith("root_") and file.endswith(".pkl"):
            os.remove(file)

if __name__ == "__main__":
    # clean all
    remove_generated_files()
