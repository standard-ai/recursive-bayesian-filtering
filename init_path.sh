#!/bin/bash
# For local development. Lets all packages in subdirectories see each other.
# Run using `$ . ./path_init.sh`.
project_directory=`pwd`
export PYTHONPATH=$PYTHONPATH:$project_directory