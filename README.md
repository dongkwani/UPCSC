# Unlocking the Potential of Unlabeled Data in Semi-Supervised Domain Generalization

This code is the official implementation of the following paper: [Unlocking the Potential of Unlabeled Data in Semi-Supervised Domain Generalization](https://arxiv.org/abs/2503.13915). The paper addresses a practical yet under-explored challenge in semi-supervised domain generalization: how to effectively utilize unconfident unlabeled data that existing methods typically discard. While current approaches focus on confident pseudo-labels, they overlook valuable learning signals in uncertain predictions. Our approach, UPCSC, introduces two novel modules: 1) Unlabeled Proxy-based Contrastive Learning (UPC) that treats unconfident samples as informative negative pairs, and 2) Surrogate Class Learning (SC) that generates positive pairs from confusion patterns. This plug-and-play framework integrates seamlessly into existing methods without requiring domain labels. Experiments on four benchmarks demonstrate consistent improvements in accuracy and generalization.

## Installation

The code is developed from [StyleMatch](https://github.com/KaiyangZhou/Dassl.pytorch), which is built on framework [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).
```
cd Dassl.pytorch/

pip install -r requirements.txt

python setup.py develop
```

You also need to prepare the weight of the decoder and the VGG for [AdaIN](https://arxiv.org/abs/1703.06868), which is used for style augmentation. Download the weights from https://github.com/naoto0804/pytorch-AdaIN and prepare them under the folder `ssdg-benchmark/weights`.

## Dataset Preperation

We support datasets PACS, OfficeHome, DigitDG, and miniDomainNet in this repository. Please download the datasets from the official links, and follow dataset structure below. You have to provide the location of root directory of datasets in the shell scripts which can be found in `scripts/`, as python parameter '--root'.

```
$DATA/
|–– pacs/
|   |–– images/
|   |–– splits/
|   |–– splits_ssdg/
|–– office_home_dg/
|   |–– art/
|   |–– clipart/
|   |–– product/
|   |–– real_world/
|   |–– splits_ssdg/
|-- digits_dg/
|   |-- mnist/
|   |-- mnist_m/
|   |-- svhn/
|   |-- syn/
|   |–– splits_ssdg/
|-- domainnet/
|   |-- clipart/
|   |-- image_list/
|   |-- painting/
|   |-- real/
|   |-- sketch/
|   |–– splits_mini_ssdg/
```

The labeled-unlabeled splits (5 per class & 10 per class) can be downloaded at the following links: [pacs](https://drive.google.com/file/d/1PSliZDI9D-_Wrr3tfRzGVtN2cpM1892p/view?usp=sharing), [officehome](https://drive.google.com/file/d/1hASLWAfkf4qj-WXU5cx9uw9rQDsDvSyO/view?usp=sharing), [digitsdg](https://drive.google.com/file/d/1ltgwO_HMnv9UudYmk3IfTaEUtDa2dLNF/view?usp=sharing), [minidomainnet](https://drive.google.com/file/d/1j7tdAXH-AWH5HmO9L0wPrRw9mw1zI9lW/view?usp=sharing). The splits need to be extracted and placed to each dataset folders' `splits_ssdg/` directory (`splits_mini_ssdg` for minidomainnet). We provide five random splits used in the experiment, and if more splits are needed, more splits can be easily created by random selection.


## Run UPCSC

```
# PACS dataset
./scripts/UPCSC.sh ssdg_pacs

# OH dataset
./scripts/UPCSC.sh ssdg_officehome

# DigitsDG dataset
./scripts/UPCSC.sh ssdg_digitdg

# miniDomainNet dataset
./scripts/UPCSC.sh ssdg_minidomainnet
```
You can also run ERM and StyleMatch method using `ERM.sh` and `StyleMatch.sh`.

For the **CFG** parameter, v1 refers to StyleMatch and v4 refers to FixMatch in script `StyleMatch.sh`.
In `UPCSC.sh`, v1 refers to UPCSC applied on StyleMatch and v4 refers to UPCSC applied on FixMatch.

The **NLAB** parameter specifies how many labeled data points to use, and the NLAB values corresponding to 5 and 10 per class for each dataset are set as follows.

| `Dataset` | 10 per class NLAB | 5 per class NLAB |
|---|---|---|
|`ssdg_pacs`| 210 | 105 |
|`ssdg_officehome`| 1950 | 975 |
|`ssdg_digitdg`| 300 | 150 |
|`ssdg_minidomainnet`| 3780 | 1890 |


## Citation
```
@inproceedings{lee2025unlocking,
  title={Unlocking the Potential of Unlabeled Data in Semi-Supervised Domain Generalization},
  author={Lee, Dongkwan and Hwang, Kyomin and Kwak, Nojun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30599--30608},
  year={2025}
}
```
