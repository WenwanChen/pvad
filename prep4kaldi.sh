#!/usr/bin/env bash
# ~~~~~~~~~
# prep4kaldi.sh
# ~~~~~~~~~
# sh prep4kaldi.sh '/dataroot/train-clean-100' textgrid random

# This script helps data preparation for building an ASR system in Kaldi
# by creating 'text', 'utt2spk', 'segments', and 'wav.scp' under 'required' folder.

# Inputs to be specified: 
# (1) datadir
#    - Directory path of where subfolders named by speaker ids are located.
#    e.g. Given /Users/cho/mycorpus/,
#                ├─ s01/            
#                ├─ s02/        NB. each subfolder includes
#                ├─ s03/            its corresponding speaker's
#                ├─ ...            -> recordings (.wav)
#                ├─ s19/            -> textgrids (.TextGrid)
#                └─ s20/
#
#        Specify as:
#            $ datadir='/Users/cho/mycorpus/'
#
# (2) datatype
#   - Type of data from which information should be extracted.
#   - Please choose between 'textgrid' or 'wavtxt'.
#
# (2) tiername
#    - Name of TextGrid tier to extract labels from (if datatype is specified as 'textgrid').
#    e.g. 'utterance', 'sent', ...
#
# Usage: $ sh prep4kaldi.sh <datadir> <datatype> <tiername>

# Created: 2017-02-27
# Last updated: 2017-02-27

# Yejin Cho (scarletcho@gmail.com)
# ─────────────────────────────────────────────────────────────────────────
# Input section
datadir=$1
datatype=$2
tiername=$3


Data clean-up
if [ -d $datadir/tmp ]; then rm -rf $datadir/tmp; fi
mkdir $datadir/tmp

if [ -e $datadir/uttinfo.txt ]; then mv $datadir/uttinfo.txt $datadir/tmp; fi
if [ -d $datadir/required/ ]; then mv $datadir/required/ $datadir/tmp/prev_required/; fi

# Delete tmp folder if empty
if [ -z "$(ls -A $datadir/tmp)" ]; then rm -rf $datadir/tmp; fi


STEP1 read textgrid, produce 10ms labels for a specific spk
case $datatype in
    textgrid)
        echo '[STEP1] Extract uttinfo.txt from .TextGrids in $datadir'
        if [ -n "$tiername" ]; then
            # When extracting info from TextGrids:
            python frameLabels.py "$datadir"
        else
            echo "[InputError] Please specify your tiername."
            exit 1
        fi
        ;;

    wavtxt)
        echo '[STEP1] Extract uttinfo.txt from .txts & .wavs in $datadir'
        # When extracting info from wavs and txts:
            python wavtxt2info.py "$datadir"
        ;;

    *)
        echo "[InputError] Please specify your datatype as 'textgrid' of 'wavtxt'."
        exit 1
        ;;
esac

STEP2 summarize all the 10ms labels for an utt



# python frame_labels_spk.py "$speaker_name"
# echo 'summarized individual labels for $speaker_name'

# at /dataroot/train-other-500 dir, do:
find . -type d -maxdepth 1 > /dataroot/prep4kaldi/train-other-500.txt
sed 's/.\///' /dataroot/prep4kaldi/train-clean-360.txt > /dataroot/prep4kaldi/train-clean-360-spk.txt

input="/dataroot/prep4kaldi/dev-clean-spk.txt"
while IFS= read -r line
do
#   echo "$line"
python frame_labels_spk.py "$line"
# echo 'summarized individual labels for' $line
done < "$input"

