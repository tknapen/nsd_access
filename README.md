# nsd_access

This package provides a single class (`NSDAccess`) allowing the user to quickly and easily access the data from the Natural Scenes Dataset.

It provides, in arbitrary volume or surface-based formats:
- one-line access to regions of interest and mapper experiment results
- one-line access to arbitrary trial betas (of any provided type)
- one-line access to all behavioral output for arbitrary trials
- one-line access to all images in the dataset, and
- one-line access to the COCO annotations of all images in the dataset.


For more information on this dataset and the project generating it, see [the NSD project website](http://naturalscenesdataset.org)


### Requirements

Apart from the 'standard' nibabel, numpy, pandas and h5py, [pycocotools](https://github.com/cocodataset/cocoapi) needs to be installed.

[![DOI](https://zenodo.org/badge/209246466.svg)](https://doi.org/10.5281/zenodo.14165748)
