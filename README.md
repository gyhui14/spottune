# SpotTune: Transfer Learning through Adaptive Fine-tuning
![spottune](https://user-images.githubusercontent.com/46233502/50563340-06dd6800-0d57-11e9-9d97-476167ba1f47.png)

This is the code accompanying the work:
Yunhui Guo, Honghui Shi, Abhishek Kumar, Kristen Grauman, Tajana Rosing, Rogerio Feris. SpotTune: Transfer Learning through Adaptive Fine-tuning [[arxiv]](https://arxiv.org/abs/1811.08737)  

## Abstract

Transfer learning, which allows a source task to affect the inductive bias of the target task, is widely used in computer vision. The typical way of conducting transfer learning with deep neural networks is to fine-tune a model pretrained on the source task using data from the target task. Current methods employ a global fine-tuning strategy, i.e., the decision of which parameters to freeze vs finetune is taken for all the examples in the target task. The assumption is that such a decision is optimal for the entire target data distribution, which may not be true, particularly in the case of insufficient target training data. We propose a novel method that decides, per training example, which layers of the pre-trained model should have their parameters fixed, i.e., shared with the source task, and which layers should be fine-tuned to improve the accuracy of the model in the target domain. Our method achieves the highest score across the board on the Visual Decathlon datasets. 

## Requirements
- Python (2.7)
- PyTorch (0.4.1)
- COCO API (from https://github.com/cocodataset/cocoapi)

**Pretrained models**:
Pretrained ResNet26 can be found at https://drive.google.com/file/d/1fiFyfb9f3PqVI4q26tp4bP9yNOFXS1KG/view?usp=sharing

**Datasets**: 
The datasets can be downloaded by following the steps listed at https://www.robots.ox.ac.uk/~vgg/decathlon/#download. Or you can download the dataset using download_data.sh which is originated from https://github.com/srebuffi/residual_adapters. If you use download_data.sh, download the data with `download_data.sh ./data/`. Then copy decathlon_mean_std.pickle to the data folder.

## Training models
To train the models on all the Visual Decathlon datasets, run `python main.py`. This will generated the trained models for all the datasets and save the trained model in ./cv. 

## Submit the results
For submiting the validation results, run `python submit_val.py`. For submiting the test results, run `python submit_test.py`. Each script generates a file called `results.json`, compress and upload the file by following the steps on https://www.robots.ox.ac.uk/~vgg/decathlon/.

## Cite
If you find this repository useful in your own research, please consider citing:
```
@article{guo2018spottune,
  title={SpotTune: Transfer Learning through Adaptive Fine-tuning},
  author={Guo, Yunhui and Shi, Honghui and Kumar, Abhishek and Grauman, Kristen and Rosing, Tajana and Feris, Rogerio},
  journal={arXiv preprint arXiv:1811.08737},
  year={2018}
}
```

