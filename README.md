# WholeBodyATQuantification
Supplementary code for the quantification of whole-body adipose tissue compartments and ectopic fat depots from whole-body Dixon MRI.

I automated to pipeline from whole-body MR images in DICOM format aquired in multiple overlapping blocks to adipose tissue compartment volumes, ectopic fat percentages and more using the following packages:
1. Conversion from DICOM to nifti: [dicom2nifti](https://pypi.org/project/dicom2nifti/)
2. Stitching image blocks into one whole-body data set: [3D Slicer](https://www.slicer.org/) and [SlicerStitchImageVolumes](https://github.com/mikebind/SlicerStitchImageVolumes)
3. Segmentation of adipose tissue compartments and ectopic fat deposits: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - trained model weights are available upon collaboration request
4. Quantification of volumes and fat percentages using functions provided in [quantification_from_segmentation.py](quantification_from_segmentation.py)
