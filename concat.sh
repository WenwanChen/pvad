#!/usr/bin/env bash

rm -rf 10ms_labels_copy/
mkdir 10ms_labels_copy/
cp -r 10ms_labels/* 10ms_labels_copy/
python concat.py