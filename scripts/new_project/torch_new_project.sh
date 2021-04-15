#!/bin/bash
# Usage: torch_new_project.sh $1
#
# This script initializes new pytorch project with the template files.
#
# Add this line to ~/.bashrc
# `export Pytorch_Template=/path/to/Pytorch_Template`
# Add symbolic link at /usr/local/bin so that you can run this script everywhere.
# `sudo ln -s $Pytorch_Template/scripts/torch_new_project.sh /usr/local/bin/torch_new_project`

source_dir=$Pytorch_Template
target_dir=$1

# copy template files
rsync -av --exclude-from="$source_dir/scripts/new_project/copy_exclude" $source_dir/ $target_dir
if [ $? -eq 0 ] # $?: exit status of last executed command
then
    echo "New project initialized at $target_dir"
else
    echo "Error while running rsync."
fi
