# CostDCNet

This repository contains the accompanying code for [CostDCNet: Cost Volume based Depth Completion for a Single RGB-D Image, ECCV'22](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620248.pdf)

## Overview

Successful depth completion from a single RGB-D image requires both extracting plentiful 2D and 3D features and merging these heterogeneous features appropriately. 
We propose a novel depth completion framework, CostDCNet, based on the cost volume-based depth estimation approach that has been successfully employed for multi-view stereo (MVS). 
The key to high-quality depth map estimation in the approach is constructing an accurate cost volume. To produce a quality cost volume tailored to single-view depth completion, we present a simple but effective architecture that can fully exploit the 3D information, three options to make an RGB-D feature volume, and a per-plane pixel shuffle for efficient volume upsampling.
Our framework consists of lightweight (~1.8M parameters) deep neural networks, running in real time (~30ms). Nevertheless, thanks to our simple but effective design, CostDCNet demonstrates depth completion results comparable to or better than the state-of-the-art (SOTA) methods.

## Getting Started

### Prerequisites

- Ubuntu 18.06 or higher
- CUDA 11.1 or higher
- pytorch 1.8 or higher
- python 3.8 or higher


### Environment Setup (Anaconda)
We recommend using Anaconda
```
conda create -n costDCNet python==3.8.12
conda activate costDCNet
conda install openblas-devel -c anaconda
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install -r requirements.txt
```

### Testing (NYUv2)
We used preprocessed NYUv2 dataset like [NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20).
```
python eval_nyu.py --data_path PATH_TO_NYUv2
```

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## Useful Links
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

## Citing CostDCNet
```
@inproceedings{kam2022costdcnet,
  title={CostDCNet: Cost Volume Based Depth Completion for a Single RGB-D Image},
  author={Kam, Jaewon and Kim, Jungeon and Kim, Soongjin and Park, Jaesik and Lee, Seungyong},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part II},
  pages={257--274},
  year={2022},
  organization={Springer}
}
```

## Related projects

**NOTE** : Our implementation is based on the repositories as:
- [Minkowski Engine, a neural network library for sparse tensors](https://github.com/StanfordVL/MinkowskiEngine)
- [Digging into Self-Supervised Monocular Depth Prediction, ICCV'19](https://github.com/nianticlabs/monodepth2)
- [Non-Local Spatial Propagation Network for Depth Completion, ECCV'20](https://github.com/zzangjinsun/NLSPN_ECCV20)
