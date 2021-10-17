# Learning with Noisy Labels for Robust Point Cloud Segmentation (ICCV2021 Oral)

![](imgs/fig1.png)



### [Project Page](https://shuquanye.com/PNAL_website/) | [Paper (ArXiv)](https://arxiv.org/abs/2107.14230) | [Pre-trained Models]() | [Supplemental Material]()



**This repository is the official pytorch implementation of the proposed *Point Noise-Adaptive Learning (PNAL)* framework our ICCV 2021 oral paper, *Learning with Noisy Labels for Robust Point Cloud Segmentation*.**

[Shuquan Ye](https://shuquanye.com/)<sup>1</sup>,
[Dongdong Chen](https://www.dongdongchen.bid/)<sup>2</sup>,
[Songfang Han](http://hansf.me/)<sup>3</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>1</sup> <br>
<sup>1</sup>City University of Hong Kong, <sup>2</sup>Microsoft Cloud AI,<sup>3</sup> University of California

## :star2: Pipeline
![](imgs/pipeline.png)


## :ok_hand: Updates 

2021/10/17: initial release.

## Install

### :point_right: Requirements

`Ubuntu 18.04`
`Conda` with `python = 3.7.7`
`pytorch = 1.5.0`
`cuda = 10.1, cudnn = 7.6.3`
`torchvision = 0.6.0`
`torch_geometric = 1.6.1`

By default, we train with a single GPU >= 10000MiB, with batchsize=12


## Refined Dataset

### :star2: Download scannet annotation refined by us. :star2:

:sparkles: [Refined Annotation](https://portland-my.sharepoint.com/:u:/g/personal/shuquanye2-c_my_cityu_edu_hk/EbG34VF22klAt1f5PK-0w10BC_8A8puqs89q9ASvaEN5Qw?e=kqcvly) :sparkles:

Download and unzip.

### Extract point clouds from ScanNetV2 raw data. 

Note that point cloud data is NOT included in the above file, according to [ScanNet Terms of Use](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf).
Download all meshes from [ScannNetv2](http://www.scan-net.org/changelog#scannet-v2-2018-06-11) validation set to `mesh/`.
And then extract by
``` python find_rgb.py ```

## Data Preparation

### :clap: Noisy and cluster data prepared by us 

Download [per60_0.018_DBSCANCluster](https://portland-my.sharepoint.com/:u:/g/personal/shuquanye2-c_my_cityu_edu_hk/EXGIfevK69xNo7QNXedClBIBMNzcb6TelxiTvbQUIv23Eg?e=OlsOHu), the S3DIS dataset with 60% symmetric noise and clustered by DBSCAN.

Move it to `NL_S3DIS/` and unzip.

### :walking: Make noisy and cluster data on your own? 

TODO

~~download and unzip data_raw.zip, the clean data and based on this we make noise.~~

~~e.g. create 60% symmetric noise:~~

```
​```python make_NL_S3DIS.py --noiserate_percent 60 --alpha 0.85 --root data_with_ins_label```
```

~~You can further switch cluster methods and noise types in S3DIS_instance.~~

~~download ply_data_all_h5, the raw S3DIS dataset.~~

~~move it to NL_S3DIS/raw and unzip.~~

~~done.~~


### :punch: How can I check the noise rate?

Go into `NL_S3DIS/` and run

``` python compare_labels.py ```

. Be patient and wait for it end to print Overall Noise Rate for you.

## Run

### :fire: PNAL 

You can run under ours PNAL pipeline with different configs, by:

```bash run_pnal.sh```

e.g., run DGCNN on S3DIS with 60% symmetric noise in our prepared `configs/PNAL.yaml`.

### :heart: without PNAL 

You can run without ours PNAL pipeline with different configs, by:

```bash run.sh```

e.g., run DGCNN on S3DIS with Symmetric Cross Entropy (SCE) Loss in our prepared `configs/SCE.yaml`, and
you can run with common Cross Entropy (CE) loss or Generalized Cross Entropy (GCE) Loss by change `LOSS_FUNCTION` from `SCE` to `""` or `GCE`...


## BibTeX
```
@article{pnal2021,
  author    = {Ye, Shuquan and Chen, Dongdong and Han, Songfang and Liao, Jing},
  title     = {Learning with Noisy Labels for Robust Point Cloud Segmentation},
  journal   = {International Conference on Computer Vision},
  year      = {2021},
}
```


## :smiley_cat: Acknowledgements 

:smile_cat: We thank a lot for the flexible codebase of [SELFIE](https://github.com/kaist-dmlab/SELFIE/blob/master/SELFIE/algorithm/selfie.py), [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/dgcnn_segmentation.py), [Truncated-Loss](https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py).

:kissing_cat: I would like to give my particular thanks to [Jiaying Lin](https://jiaying.link/), my special friend, for his constructive suggestions, generous support to this project, as well as the tremendous love given to me~
