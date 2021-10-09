#!/bin/sh
# Extract the BioNLP 2019 `BB-Rel` dataset.
clean_and_extract() {
    [ -d "./BioNLP-2019/bb-rel/BioNLP-OST-2019_BB-rel_$1" ] && rm -rf "./BioNLP-2019/bb-rel/BioNLP-OST-2019_BB-rel_$1"
    unzip -d './BioNLP-2019/bb-rel' "./BioNLP-2019/bb-rel/BioNLP-OST-2019_BB-rel_$1.zip"
}

clean_and_extract 'dev'
clean_and_extract 'train'
clean_and_extract 'test'
