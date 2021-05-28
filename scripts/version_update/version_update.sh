#!/bin/bash
# Merge files from old version to new template version
# bash -x ./scripts/version_update/version_update.sh $1
# $1: ../new_version_dir/
rsync -arv --exclude='__pycache__' --files-from=scripts/version_update/preserved_files . $1
# note
