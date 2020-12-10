#!/bin/bash
# Usage: ./torch_new_project.sh $1
#
# This script initializes new pytorch project with the template files.
#
# Add the following lines to ~/.bashrc so that you can run this script everywhere.
# export Pytorch_Template=/path/to/Pytorch_Template
# export PATH=$PATH:$Pytorch_Template

export source_dir=$Pytorch_Template
export target_dir=$1

# copy template files
rsync -av --exclude-from="$source_dir/copy_exclude" $source_dir/ $target_dir
if [ "$?" -eq "0" ]
then
    # remove first line in __init__.py
    for FILE in "$target_dir/data_loader/__init__.py" "$target_dir/model/__init__.py" "$target_dir/trainer/__init__.py"
    do
        tail -n +2 "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
    done
    echo "New project initialized at $target_dir"
else
    echo "Error while running rsync."
fi
