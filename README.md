# Schizophrenia classification on image dataset  

**schizoCNN** is a general purpose library for learning-based tools for classification on existing Deep learning architecture on keras application.

# Tutorial

contact to [Schizo_CNN developer](https://github.com/yoonguusong) to learn about this algorithm


# Instructions

To use the Schizo_CNN library, either clone this repository and install the requirements listed in `setup.py` ~~or install directly with pip.~~

## Preprocessing
<img src="./data/data_example_healthy.jpg" width="50%"> <img src="./data/data_example_schizophrenia.jpg" width="50%">

<p>
    <img src="./data/data_example_healthy.jpg" width = "300" alt>
    <em>healthy subject data</em>
</p>

## Training

If you would like to train your own model, you will likely need to customize some of the data loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you have a directory containing training data files in npz (numpy) format. It's assumed that each npz file in your data folder has a `vol` parameter, which points to the numpy image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation (for semi-supervised learning). It's also assumed that the shape of all image data in a directory is consistent.

For a given `/path/to/training/data`, the following script will train the dense network (described in MICCAI 2018 by default) using scan-to-scan registration. Model weights will be saved to a path specified by the `--model-dir` flag.

```
./scripts/tf/train.py /path/to/training/data --model-dir /path/to/models/output --gpu 0
```

Scan-to-atlas registration can be enabled by providing an atlas file with the `--atlas atlas.npz` command line flag. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.


## Testing (measuring Dice scores)

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
./scripts/tf/test.py --model model.h5 --atlas atlas.npz --scans scan01.npz scan02.npz scan03.npz --labels labels.npz
```

Just like for the training data, the atlas and test npz files include `vol` and `seg` parameters and the `labels.npz` file contains a list of corresponding anatomical labels to include in the computed dice score.



# Contact:
For any problems or questions please contact email <syg7949@naver.com>  
