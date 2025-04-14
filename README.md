# ID-PML
Instance-Dependent Partial Multi-Label Learning via Decomposition of Candidate Labels Generation

Then directly run the following command to generate json file (for implementation) of these datasets.
```
python utils/prepare/prepare_voc.py  --data_path  Dataset/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  Dataset/COCO2014
```
Please note that the generated json file is initially in `utils/prepare`, please move it to `data/voc12` manually.
