# U-GAT-IT &mdash; with Face Recognition Loss

## Fork メモ

UGATIT の Generator ロスは次の 5 つ

- Adversarial loss
  - s An adversarial loss is employed to match the distribution of the translated images to the target image distribution
- Cycle loss (Reconstruction loss)
  - To alleviate the mode collapse problem, we apply a cycle consistency constraint to the generator
- Identity loss
  - To ensure that the color distributions of input image and output image are similar
- CAM loss
  - exploiting the information from the auxiliary classifiers ηs and ηDt

実写 → イラストの場合に限り、変換前後で人が変わっているかどうかのロスを追加することができると考えた。もとのモデルでは 猫 → 犬 のように全く異なるものへの変換も想定しているため、このようなロスは存在しない。

- Face distance loss
  - A->B, B->A の前後の FaceRecognition モデルの一致度をロスとして利用する、FaceRecognition モデルは `trainable=False` とする

## 使い方

基本的な使い方はフォーク元のリポジトリを参照してください。

### Face Recognition Loss について

ロス関数作成については[Facenet](https://github.com/davidsandberg/facenet)を使用しました。

README にある学習済みモデルを[ダウンロード](https://github.com/davidsandberg/facenet#pre-trained-models)し、実行時に `--facenet_checkpoint_dir` で指定してください。指定しない場合はフォーク元と同じ挙動を示すはずです。

また、使用する GPU によっては `img_size` がデフォルトの 256 だとメモリが足りなくなる場合があります。その際には `--img_size 128` オプションをつけることで動作する可能性があります。

例

```
$ python3 main.py --facenet_checkpoint_dir facenet_weight/20180408-102900 --img_size 128
```

また、Face Distance Loss の重みは `--face_distance_weight` で指定することができます。使用するデータセットに合わせて調整してください。

学習途中の Face Distance Loss は Tensorboard 上の `G_X_face_distance` で確認することができます。

```
$ tensorboard --logdir logs
```

### : Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

<div align="center">
  <img src="./assets/teaser.png">
</div>

### [Paper](https://arxiv.org/abs/1907.10830) | [Official Pytorch code](https://github.com/znxlwm/UGATIT-pytorch)

This repository provides the **official Tensorflow implementation** of the following paper:

> **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br> > **Junho Kim (NCSOFT)**, Minjae Kim (NCSOFT), Hyeonwoo Kang (NCSOFT), Kwanghee Lee (Boeing Korea)
>
> **Abstract** _We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters._

## Requirements

- python == 3.6
- tensorflow == 1.14

## Pretrained model

> We released 50 epoch and 100 epoch checkpoints so that people could test more widely.

- [selfie2anime checkpoint (50 epoch)](https://drive.google.com/file/d/1V6GbSItG3HZKv3quYs7AP0rr1kOCT3QO/view?usp=sharing)
- [selfie2anime checkpoint (100 epoch)](https://drive.google.com/file/d/19xQK2onIy-3S5W5K-XIh85pAg_RNvBVf/view?usp=sharing)

## Dataset

- [selfie2anime dataset](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing)

## Web page

- [Selfie2Anime](https://selfie2anime.com) by [Nathan Glover](https://github.com/t04glovern)
- [Selfie2Waifu](https://waifu.lofiu.com) by [creke](https://github.com/creke)

## Telegram Bot

- [Selfie2AnimeBot](https://t.me/selfie2animebot) by [Alex Spirin](https://github.com/sxela)

## Usage

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg
           ├── ddd.png
           └── ...
```

### Train

```
> python main.py --dataset selfie2anime
```

- If the memory of gpu is **not sufficient**, set `--light` to **True**
  - But it may **not** perform well
  - paper version is `--light` to **False**

### Test

```
> python main.py --dataset selfie2anime --phase test
```

## Architecture

<div align="center">
  <img src = './assets/generator_fix.png' width = '785px' height = '500px'>
</div>

---

<div align="center">
  <img src = './assets/discriminator_fix.png' width = '785px' height = '450px'>
</div>

## Results

### Ablation study

<div align="center">
  <img src = './assets/ablation.png' width = '438px' height = '346px'>
</div>

### User study

<div align="center">
  <img src = './assets/user_study.png' width = '738px' height = '187px'>
</div>

### Kernel Inception Distance (KID)

<div align="center">
  <img src = './assets/kid_fix2.png' width = '750px' height = '400px'>
</div>

## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{
Kim2020U-GAT-IT:,
title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwang Hee Lee},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlZ5ySKPH}
}
```

## Author

[Junho Kim](http://bit.ly/jhkim_ai), Minjae Kim, Hyeonwoo Kang, Kwanghee Lee
