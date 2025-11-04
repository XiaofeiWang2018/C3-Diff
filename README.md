# C3-Diff: Super-resolving Spatial Transcriptomics via Cross-modal Cross-content Contrastive Diffusion Modelling

## 1. Environment
- Python >= 3.8
- Pytorch >= 2.0 is recommended
- opencv-python
- sklearn
- matplotlib


## 2. Train
Use the below command to train the model on Xenium [[Data Link]](https://huggingface.co/datasets/Zeiler123/C3-Diff/resolve/main/Xenium.zip) based on the [[pretrained weights]](https://huggingface.co/datasets/Zeiler123/C3-Diff/resolve/main/toy-uncon.ckpt).
. 
```
    python ./train_ours.py
```

## 3. Test
Use the below command to test the model on the database.
```
    python ./test_ours.py
```




