# pvad
speaker conditioned voice activity detection
https://arxiv.org/abs/1908.04284

![image](https://user-images.githubusercontent.com/41505580/143957774-6b68790e-9752-44b2-bc69-c6c0307810a5.png)

Classifier: {non-speech, target speaker, and non-target speaker}

1. Synthetic dataset generation\
    prep4kaldi.sh  \
    flac_to_wav.sh \
    concat.sh concat.py \
    augment.py 

2. Prepare target speaker embeddings\
    extract_embeddings.py

3. Extract features and labels\
    correct_target_labels.py\
    fbank.py\
    feature_labels.py

4. Data loader\
    dataloader.py\
    dataloader_test.py


5. Model definition and traning\
    pvad_training.py
