#!/bin/bash

#pip install gdown
# Download the embeddings
#gdown https://drive.google.com/uc?id=1F70CtbsoPPPDnV-ZAUq0i0Rrvtv6taoV
#sudo apt-get install unzip
#unzip CodenamesData.zip
#rm CodenamesData.zip
#mv CodenamesData data

#download conceptnet
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sE6NFOZy_q9dH0obGW29BEzNGrcUYcd3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sE6NFOZy_q9dH0obGW29BEzNGrcUYcd3" -O conceptnet-assertions-en && rm -rf /tmp/cookies.txt

#download numberbatch
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=17Bs7zfjOxGQfjhHtBvTmR88o6m8Ecr4d' -O 'numberbatch-small.txt'