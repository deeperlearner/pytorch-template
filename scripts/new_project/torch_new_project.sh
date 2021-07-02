#!/usr/bin/env bash
# Usage: torch_new_project.sh $1
#
# This script initializes new pytorch project with the template files.
#
# Add this line to ~/.bashrc
# `export pytorch_template=/path/to/pytorch-template`
# Add symbolic link at /usr/local/bin so that you can run this script everywhere.
# `sudo ln -s $pytorch_template/scripts/new_project/torch_new_project.sh /usr/local/bin/torch_new_project`

source_dir=$pytorch_template
target_dir=$1

# copy template files
rsync -av --exclude-from="$source_dir/scripts/new_project/copy_exclude" $source_dir/ $target_dir
if [ $? -eq 0 ] # $?: exit status of last executed command
then
    echo "New project initialized at $target_dir"
else
    echo "Error while running rsync."
fi
