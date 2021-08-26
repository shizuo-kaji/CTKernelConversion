Python code for converting sharp-kernel reconstructed CT images to soft-kernel images
=============

This is a companion code for the paper
- *Kernel conversion for chest-computed tomography using deep learning-based image-to-image translation* 
by Naoya Tanabe, Shizuo Kaji, Hiroshi Shima, Yusuke Shiraishi, Tsuyoshi Oguma, Susumu Sato, Toyohiro Hirai

The code is based on 
- [Image-to-image translation by CNNs trained on paired data](https://github.com/shizuo-kaji/PairedImageTranslation) which is described in the paper
- [*Overview of image-to-image translation using deep neural networks: denoising, super-resolution, modality-conversion, and reconstruction in medical imaging*](https://link.springer.com/article/10.1007/s12194-019-00520-y)
by Shizuo Kaji and Satoshi Kida, Radiological Physics and Technology,  Volume 12, Issue 3 (2019), pp 235--248, [arXiv:1905.08603](https://arxiv.org/abs/1905.08603)

### Requirements
- python 3: [Anaconda](https://anaconda.org) is recommended
- chainer >= 7.2.0, chainercv, opecv, pydicom: install them by
```
pip install -U git+https://github.com/chainer/chainer.git
pip install -U chainercv opencv-contrib-python pydicom
```

(optional, but recommended)
To use GPU, you have to install CUDA and CuPy.
- CUDA: follow the instruction [here](https://docs.nvidia.com/cuda/index.html)
- CuPy: follow the instruction [here](https://cupy.dev)


### How to use
- Arrange DICOM files into the directory structure as in the following example:
(phantom images are included as demo)
```
sharp_kernel
+-- patient1
|   +-- patient1_001.dcm
|   +-- patient1_002.dcm
|   +-- ...
+-- patient2
|   +-- patient2_001.dcm
|   +-- patient2_002.dcm
|   +-- ...
```
- Execute the following commands from the terminal/command prompt:
```
    python convert.py -a args_partial -R sharp_kernel -o partial
    python convert.py -a args_full -R sharp_kernel -o full
    python dicom_overlay.py -i0 sharp_kernel -i1 full -i2 partial -o converted
```
- The first line creates a (temporary) directory named *partial* which contains converted images by the partial model (that is, CT values are cropped to -300 to 300 HU).
- The second line creates a (temporary) directory named *full* which contains converted images by the full model.
- The third line creates a directory named *converted* which contains the final converted images obtained by fusing the above two.


## Licence
MIT Licence

