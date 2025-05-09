# Instance-Dependent Partial Multi-Label Learning via Decomposition of Candidate Labels Generation (DGCL)

## Dataset
To evaluate/train DGCL, first, the [VOC2012/2007](http://host.robots.ox.ac.uk/pascal/VOC/) and [MSCOCO](https://cocodataset.org/#download) datasets need to be downloaded.

Then directly run the following command to generate json file (for implementation) of these datasets.
```
python utils/prepare/prepare_voc07.py  --data_path  Dataset/VOCdevkit
python utils/prepare/prepare_voc12.py --data_path Dataset/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  Dataset/COCO2014
```
Please note that the generated json file is initially in `utils/prepare`, please move it to `data/voc12` manually. In addition, for VOC2012 dataset, since the test set is VOC2007, please directly move the `test_voc07.json` generated in voc07 to `data/voc12` without renaming.

## Noise Generation
In order to provide a fair comparison environment, we placed the various three noise labeling environments of the constructed VOC2007 and MS-COCO datasets in `pre-processed-data/`.
### 1. Part-dependent IDN Generation
```
pre-processed-data/coco_0.1.npy
...
pre-processed-data/coco_0.5.npy
pre-processed-data/voc07_0.1.npy
...
pre-processed-data/voc07_0.5.npy
pre-processed-data/voc12_0.1.npy
...
pre-processed-data/voc12_0.5.npy
```
### 2. Classification-based IDN Generation
In order to provide a fair comparison environment, we placed the various three noise labeling environments of the constructed VOC2007 and MS-COCO datasets in `pre-processed-data/`.
For example, in the MS-COCO dataset for Type-I with a noise rate of 0.35, as follows:
```
pre-processed-data/coco_1_0.35.npy
```
You can also generate your own `Classification-based IDN Generation` by running the `gen_noise_label.py`:
```
python -u gen_noise_label.py --dataset coco --img_size 224 --partial_rate 0.35 --noise_type 1 --total_epoch 5
python -u gen_noise_label.py --dataset voc07 --img_size 224 --partial_rate 0.35 --noise_type 1 --total_epoch 10
python -u gen_noise_label.py --dataset voc12 --img_size 224 --partial_rate 0.35 --noise_type 1 --total_epoch 10
```
## Usage
### VOC2007
If `noise_type` is part-dependent noise, it is `Direct`, otherwise it is `Indirect_1/Indirect_2/Indirect_3`.
```bash
python main_DGCL.py \
--dataset 'voc07' \
--img_size 224/448 \
--batch_size 32 \
--partial_rate 0.1-0.5/0.35/0.7 \
--noise_type Direct \
--warm_up 5 \
--batch_size 16 \
--total_epoch 30 \
```
