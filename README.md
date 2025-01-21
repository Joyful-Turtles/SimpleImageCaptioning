# Prequisite
* anaconda
* COCO Datset

## COCO Dataset download
* [train(13GB)](http://images.cocodataset.org/zips/train2014.zip)
* [validation(6GB)](http://images.cocodataset.org/zips/val2014.zip)
* [test(6GB)](http://images.cocodataset.org/zips/test2014.zip)
* [train/val annotations (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

# Setup 
```sh
conda env create -f environment.yml
```

# How to
## train
> You need to change hardcoded data COCO datapath
```sh
python train.py
```
## inference
> You need to chang hardcoded image path
```sh
python inference.py
```
