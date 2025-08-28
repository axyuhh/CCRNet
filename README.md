# CCRNet

This is the official version of CCRNet (Arbitrary style transfer via cube and cube root network and warping constraint).

0. Preparations

Download vgg_normalized.pth and put it under models/.

Download COCO2014 dataset (content dataset) and Wikiart dataset (style dataset).

2. Train

python train.py --content_dir /MSCOCO/ --style_dir /Wikiart/

2. test

python test.py --content_dir /your_content_images/ --style_dir /your_style_images/

If this code is used in your paper, please cite our paper.
