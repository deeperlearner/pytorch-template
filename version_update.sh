# Merge files from old version to new template version
# bash -x version_update.sh $1
# $1: ../new_version_dir/
rsync -arv --exclude='__pycache__' --files-from=file_list . $1
# note
