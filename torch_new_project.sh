#!/bin/bash
# Usage: torch_new_project.sh $1
#
# This script initializes new pytorch project with the template files.
#
# Add the following lines to ~/.bashrc so that you can run this script everywhere.
# export Pytorch_Template=/path/to/Pytorch_Template
# export PATH=$PATH:$Pytorch_Template

source_dir=$Pytorch_Template
target_dir=$1

# copy template files
rsync -av --exclude-from="$source_dir/copy_exclude" $source_dir/ $target_dir
if [ $? -eq 0 ] # $?: exit status of last executed command
then
    echo "New project initialized at $target_dir"
else
    echo "Error while running rsync."
fi
