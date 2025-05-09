# Instance-Dependent Partial Multi-Label Learning via Decomposition of Candidate Labels Generation (DGCL)

Then directly run the following command to generate json file (for implementation) of these datasets.
```
python utils/prepare/prepare_voc07.py  --data_path  Dataset/VOCdevkit
python utils/prepare/prepare_voc12.py --data_path Dataset/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  Dataset/COCO2014
```
Please note that the generated json file is initially in `utils/prepare`, please move it to `data/voc12` manually. In addition, for VOC2012 dataset, since the test set is VOC2007, please directly move the `test_voc07.json` generated in voc07 to `data/voc12` without renaming.

In order to provide a fair comparison environment, we placed the various three noise labeling environments of the constructed VOC2007 and MS-COCO datasets in pre-processed-data.
```
pre-processed-data/coco_0.05.npy
pre-processed-data/coco_0.1.npy
pre-processed-data/coco_0.2.npy
pre-processed-data/voc07_0.1.npy
pre-processed-data/voc07_0.2.npy
pre-processed-data/voc07_0.4.npy
```
