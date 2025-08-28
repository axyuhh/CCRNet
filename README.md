# CCRNet

This is the official version of CCRNet (Arbitrary style transfer via cube and cube root network and warping constraint).

### Preparations

Download vgg_normalized.pth and put it under models/.

Download COCO2014 dataset (content dataset) and Wikiart dataset (style dataset).

### Train

python train.py --content_dir /MSCOCO/ --style_dir /Wikiart/

### Test

To use the pre-trained models, please download here [pre-trained model](https://drive.google.com/drive/folders/1R4413DU-8-6DNoKJqj9pVeTyzrYE_yXi?usp=sharing) and put it under experiments/.

python test.py --content_dir /your_content_images/ --style_dir /your_style_images/

### Reference

The code is based on project [AdaIN](https://github.com/naoto0804/pytorch-AdaIN).

### Note

If this code is useful to you, please cite our paper.

```
@article{yu2025arbitrary,
  title={Arbitrary Style Transfer via Cube and Cube Root Network and Warping Constraint},
  author={Yu, Xiaoming and Tian, Jie and Hu, Zhenhua},
  journal={Pattern Recognition},
  pages={112314},
  year={2025},
  publisher={Elsevier}
}
```
