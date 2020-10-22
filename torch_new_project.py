#!/usr/bin/python3
import os
from os.path import expanduser, join, isfile, abspath
import sys
from shutil import copytree, ignore_patterns

# The directory of Pytorch-Template
template_dir = expanduser("~/Github/Pytorch-Template")

# This script initializes new pytorch project with the template files.
# Add `export PATH=$PATH:${HOME}/Github/Pytorch-Template` to ~/.bashrc
# so that you can run `torch_new_project.py MyNewProject` in everywhere
# and new project named MyNewProject will be made.
assert isfile(join(template_dir, 'torch_new_project.py')), 'torch_new_project.py should be in template_dir'
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: torch_new_project.py MyNewProject'

target_dir = abspath(sys.argv[1])

ignore = [".git", "data", "saved", "torch_new_project.py", "LICENSE", ".flake8", "README.md", "__pycache__"]
copytree(template_dir, target_dir, ignore=ignore_patterns(*ignore))
print('New project initialized at', target_dir)
