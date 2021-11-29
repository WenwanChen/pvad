# for i in /mnt/data0/wc43/PVAD_data/LibriSpeech/*/*/*.flac ; do 

# ffmpeg -i $i ${i%.*}'.wav'
# rm -rf $i

# done

for i in /mnt/data0/wc43/PVAD_dev/LibriSpeech/dev-clean/*/*/*.flac ; do 

ffmpeg -i $i ${i%.*}'.wav'
rm -rf $i

done