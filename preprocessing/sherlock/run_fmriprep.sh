#!/bin/bash
# fMRIPrep 25.2.3 (Docker image nipreps/fmriprep:25.2.3) on the 16 subjects of
# the Sherlock dataset (OpenNeuro ds001132 v1.0.0), with default settings: outputs in MNI152NLin2009cAsym space, written to
# <bids_dir>/derivatives/sub-XX/.
#
#   bash preprocessing/sherlock/run_fmriprep.sh /path/to/ds001132 /path/to/freesurfer/license.txt
#
# Each subject took several hours on 8 CPUs; on a cluster, submit one job per
# subject instead of running the loop.

BIDS_DIR=$1
FS_LICENSE=$2
WORK_DIR=${3:-$BIDS_DIR/../fmriprep_work}

for SUBJECT in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
    fmriprep "$BIDS_DIR" "$BIDS_DIR/derivatives" participant \
        --participant-label "$SUBJECT" \
        --fs-license-file "$FS_LICENSE" \
        --nprocs 8 --mem_mb 30000 \
        --work-dir "$WORK_DIR/sub-$SUBJECT"
done
