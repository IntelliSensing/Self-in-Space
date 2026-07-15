# VideoFlow MOFNet Subset

This directory contains the minimal VideoFlow MOFNet implementation required by
SIS-Motion's frozen optical-flow estimator. VideoFlow's standalone training,
evaluation, dataset-conversion, BOFNet, and pretrained artifact files are not
vendored here.

Upstream project: [XiaoyuShi97/VideoFlow](https://github.com/XiaoyuShi97/VideoFlow)

The default implementation uses the PyTorch correlation path. The optional CUDA
correlation extension can be compiled for the active environment:

```bash
cd motion/src/uav/external/videoflow/alt_cuda_corr
bash run_install.sh
```

Download `MOF_kitti.pth` from the upstream [pretrained models folder](https://drive.google.com/drive/folders/16YqDD_IQpzrVWvDHI9xK3kO0MaXnNIGx)
and place it at `checkpoints/VideoFlow/MOF_kitti.pth` for SIS-Motion training.

VideoFlow is released under the Apache License 2.0. Please cite the upstream
work when using this component:

```bibtex
@article{shi2023videoflow,
  title={VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation},
  author={Shi, Xiaoyu and Huang, Zhaoyang and Bian, Weikang and Li, Dasong and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and Dai, Jifeng and Li, Hongsheng},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
