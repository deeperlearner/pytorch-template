#!/usr/bin/python3
"""
This script initializes new pytorch project with the template files.
Add the following lines to ~/.bashrc :
```
export Pytorch_Template=/path/to/Pytorch_Template
export PATH=$PATH:$Pytorch_Template
```
After sourcing ~/.bashrc, you can run `torch_new_project.py NewProject` in everywhere.
"""
from os.path import expandvars, join, isfile, abspath
import sys
from shutil import copytree, ignore_patterns


# The directory path of Pytorch-Template
template_dir = expandvars("$Pytorch_Template")

assert isfile(join(template_dir, 'torch_new_project.py')), 'torch_new_project.py should be in template_dir'
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: torch_new_project.py MyNewProject'

target_dir = abspath(sys.argv[1])

ignore = [".git", "*.vim", "__pycache__", "README.md", "torch_new_project.py", "data", "saved", "*examples"]
# For Python version >= 3.6, you need to revise code in `/usr/lib/python3.6/shutil.py`, in the function of copytree()
# ```
# if not os.path.isdir(dst):
#     os.makedirs(dst)
# ```
copytree(template_dir, target_dir, ignore=ignore_patterns(*ignore))
# For Python version >= 3.8
# copytree(template_dir, target_dir, ignore=ignore_patterns(*ignore), dirs_exist_ok=True)
print('New project initialized at', target_dir)
