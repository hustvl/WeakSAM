# download vgg16 imagenet pre-trained weights
mkdir -p pretrain
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10Vh2qFmGucO-9DZ3eY3HAvcAmtPFcFg2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10Vh2qFmGucO-9DZ3eY3HAvcAmtPFcFg2" -O pretrain/vgg16.pth && rm -rf /tmp/cookies.txt

# download pascal voc selective search region proposals
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-1EJ3Mm7KoXwaurYx4zpPcerkKbIAqU-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-1EJ3Mm7KoXwaurYx4zpPcerkKbIAqU-" -O data/voc_selective_search.zip && rm -rf /tmp/cookies.txt
cd data && unzip voc_selective_search.zip && mv voc_selective_search/voc_2007* VOCdevkit/VOC2007/ && mv voc_selective_search/voc_2012* VOCdevkit/VOC2012/ && rm voc_selective_search.zip && rm -rf voc_selective_search

# download pascal voc super pixels
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tItyPAUz16iXOyIHpVfAmWbKfwQzkzoU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tItyPAUz16iXOyIHpVfAmWbKfwQzkzoU" -O data/voc_super_pixels.zip && rm -rf /tmp/cookies.txt
cd data && unzip voc_super_pixels && mv voc_super_pixels/superpixel2007 VOCdevkit/VOC2007/SuperPixels && mv voc_super_pixels/superpixel2012 VOCdevkit/VOC2012/SuperPixels && rm voc_super_pixels.zip && rm -rf voc_super_pixels

