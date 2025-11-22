# WholeBodyATQuantification
Supplementary code for the quantification of whole-body adipose tissue compartments and ectopic fat depots from whole-body Dixon MRI.

I automated to pipeline from whole-body MR images in DICOM format aquired in multiple overlapping blocks to adipose tissue compartment volumes, ectopic fat percentages and more using the following packages:
1. Conversion from DICOM to nifti: [dicom2nifti](https://pypi.org/project/dicom2nifti/)
2. Stitching image blocks into one whole-body data set: [3D Slicer](https://www.slicer.org/) and [SlicerStitchImageVolumes](https://github.com/mikebind/SlicerStitchImageVolumes)
3. Segmentation of adipose tissue compartments and ectopic fat deposits: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - trained model weights are available upon collaboration request
4. Quantification of volumes and fat percentages using functions provided in [quantification_from_segmentation.py](quantification_from_segmentation.py)

## Examples
Here are some code snippets that show how the different functions can be used to form an automated pipeline.
The output directory for the conversion part serves as input directory for the stitching part. I used a structure that looks similar to this:
```
/some/path/to/converted_block_directory/
|-- study_id_01
    |-- participant_id_01
    |  |-- fat
    |  |  |-- 00_block_1.nii.gz
    |  |  |-- 01_block_2.nii.gz
    |  |-- water
    |  |  |-- 00_block_1.nii.gz
    |  |  |-- 01_block_2.nii.gz
    |  |-- opp
    |  |  |-- 00_block_1.nii.gz
    |  |  |-- 01_block_2.nii.gz
    |-- participant_id_02
    |  |-- fat
    |  |  |-- 00_block_1.nii.gz
    |  |  |-- 01_block_2.nii.gz
    |  |-- water
    |  |  |-- 00_block_1.nii.gz
    |  |  |-- 01_block_2.nii.gz
    |  |-- opp
    |  |  |-- 00_block_1.nii.gz
    |  |  |-- 01_block_2.nii.gz
|-- study_id_02
...
```
### Conversion
DICOM folders/locations are typically very different so I skip this part. To follow the tree structure examplified above, you would need to extract your (unique) participant IDs and to identify the (sub-)folders containing to fat-selective, water-selective, and opposed-phase Dixon MR images in DICOM format (e.g., on your PACS).

```python
import os
import dicom2nifti
dicom2nifti.settings.disable_validate_slice_increment()

converted_block_directory = '/some/path/to/directory'
for study in ['study_id_01', 'study_id_02']:
    for pid in participant_ids:
        pid_sub_dir = os.path.join(converted_block_directory, study, pid, 'fat')
        if not os.path.isdir(pid_sub_dir):
            os.mkdir(pid_sub_dir)
        dicom2nifti.convert_directory('/path/to/fat-selective', pid_sub_dir)
        # Similar for water-selective and oppposed-phase images
```

### Stitching with [SlicerStitchImageVolumes](https://github.com/mikebind/SlicerStitchImageVolumes)

```python
# Setup extensions internals
stitch = slicer.util.getModuleGui("StitchVolumes")
stitch_logic = slicer.util.getModuleLogic('StitchVolumes')
pn = stitch._parameterNode

# Parameters
useThresholdValue = True
thresholdValue = -1024.0
defaultVoxelValue = 0

from stitch_image_volumes import stitch_blocks

converted_block_directory = '/some/path/to/directory'
study_names = ['Study1', 'Study2', 'Study3']

for study in study_names:
    study_sub_dir = os.path.join(converted_block_directory, study)
    pids = [i for i in os.listdir(study_sub_dir)]
    for p in pids:
        for i, dix in enumerate(['fat', 'water', 'opp']):
            block_dir = os.path.join(study_sub_dir, p, dix)
            blocks = [i for i in os.listdir(block_dir)]
            output_file = f'{p}_000{i}.nii.gz'
            stitch_blocks(blocks, block_dir, output_file)
            slicer.mrmlScene.Clear()
sys.exit(0)
```

This script is to be run inside 3D Slicers Python environment, for example, as a subprocess:
```python
import subprocess

subprocess.run(['/path/to/Slicer/Slicer.exe', '--no-main-window', '--python-script', '/path/to/script.py'])
```

### Segmentation
Install and setup a working [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) environment (following their installation instructions). After downloading and unzipping our trained segmentation model in your nnUNet_results folder, run the following command to run the segmentation. You would need fat-selective (_0000.nii.gz), water-selective (_0001.nii.gz)), and opposed-phase (_0002.nii.gz) MR images
```
nnUNetv2_predict -i /some/input/directory -o /some/output/directory -d 301 -c 3d_fullres -p nnUNetResEncUNetMPlans --continue_prediction
```
