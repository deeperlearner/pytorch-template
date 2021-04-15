# Merge files from old version to new template version
# bash -x scripts/version_update/version_update.sh $1
# $1: ../new_version_dir/
rsync -arv --exclude='__pycache__' --files-from=scripts/version_update/file_list . $1
# note
