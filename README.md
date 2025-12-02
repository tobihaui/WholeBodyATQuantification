# WholeBodyATQuantification
Supplementary code for the quantification of whole-body adipose tissue compartments and ectopic fat depots from whole-body Dixon MRI.

I automated to pipeline from whole-body MR images in DICOM format aquired in multiple overlapping blocks to adipose tissue compartment volumes, ectopic fat percentages and more using the following packages:
1. Conversion from DICOM to Nifti: [dicom2nifti](https://pypi.org/project/dicom2nifti/)
2. Stitching image blocks into one whole-body data set: [3D Slicer](https://www.slicer.org/) and [SlicerStitchImageVolumes](https://github.com/mikebind/SlicerStitchImageVolumes)
3. Segmentation of adipose tissue compartments and ectopic fat deposits: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - trained model weights are available upon collaboration request
4. Quantification of volumes and fat percentages using functions provided in [quantification_from_segmentation.py](quantification_from_segmentation.py)

## Examples
Here are some code snippets that show how the different functions can be used to form an automated pipeline.
The output directory for the conversion part serves as input directory for the stitching part. I used a structure that looks similar to:
```
/path/to/converted_block_directory/
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
...
```
### Conversion
To follow the tree structure examplified above, you would need to extract your (unique) participant IDs and to identify the (sub-)folders containing to fat-selective, water-selective, and opposed-phase Dixon MR images in DICOM format (e.g., on your PACS). On a SIEMENS system with two acquired blocks of 3D VIBE Dixon MRI, a DICOM output folder could like something similar to:
```
/path/to/directory
|-- Participant_ID
    |-- 0001_ProtocolSequence_01
    |-- 0002_ProtocolSequence_02
    |-- 0003_ProtocolSequence_03
    |-- 0004_Dixon_opp
    |-- 0005_Dixon_in
    |-- 0006_Dixon_F
    |-- 0007_Dixon_W
    |-- 0008_Dixon_opp
    |-- 0009_Dixon_in
    |-- 0010_Dixon_F
    |-- 0011_Dixon_W
    ...
```
Assuming that a unique study identifier is part of the participant_ID, a DICOM to Nifti conversion script could be based on:
```python
import os
import dicom2nifti
dicom2nifti.settings.disable_validate_slice_increment()


dcm_img_directory = '/path/to/directory'
converted_block_directory = '/path/to/converted_block_directory/'
participant_ids = [i for i in os.listdir(dcm_img_directory) if 'STUDY_ID' in i]

for pid in participant_ids:
    pid_directory = os.path.join(converted_block_directory, pid)
    if not os.path.isdir(pid_directory):
        sequences = [i for i in os.listdir(os.path.join(dcm_img_directory, pid))]
        dixon_f = [i for i in sequences if 'dixon_f' in i.lower()]
        dixon_w = [i for i in sequences if 'dixon_w' in i.lower()]
        dixon_opp = [i for i in sequences if 'dixon_opp' in i.lower()]
        # Run conversion
        pid_fat_sub_dir = os.path.join(pid_directory, 'fat')
        if not os.path.isdir(pid_fat_sub_dir):
            os.mkdir(pid_fat_sub_dir)
        for f in dixon_f:
            dicom2nifti.convert_directory(os.path.join(dcm_img_directory, pid, f),
                                          pid_fat_sub_dir)
        # Similar for water-selective and oppposed-phase images
```

### Stitching with [SlicerStitchImageVolumes](https://github.com/mikebind/SlicerStitchImageVolumes)
The stitching is based on a function provided in [stitch_image_volumes.py](stitch_image_volumes.py). You need to install [3D Slicer](https://www.slicer.org/) and should have its directory accessible from within Python. An outline of a script for the automated stitching of different Dixon blocks (if you have only one block acquired, this can be skipped) could look like:
```python
import os

from stitch_image_volumes import stitch_blocks

converted_block_directory = '/path/to/converted_block_directory/'
img_seg_directory = '/path/to/converted_image_directory'

pids = [i for i in os.listdir(converted_block_directory)]
for p in pids:
    for i, dix in enumerate(['fat', 'water', 'opp']):
        output_file = os.path.join(img_seg_directory, '{p}_000{i}.nii.gz')
        if not os.path.isfile(output_file):
            block_dir = os.path.join(converted_block_directory, p, dix)
            blocks = [i for i in os.listdir(block_dir)]
            stitch_blocks(blocks, block_dir, output_file)
            slicer.mrmlScene.Clear()
sys.exit(0)
```
Your stiching script needs to be run inside 3D Slicer's Python environment, for example, as a subprocess (inside the conversion script above):
```python
import subprocess

subprocess.run(['/path/to/Slicer/Slicer.exe', '--no-main-window', '--python-script', '/path/to/script.py'])
```
### Segmentation
Install and setup a working [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) environment (following their installation instructions). After downloading and unzipping our trained segmentation model in your nnUNet_results folder, run the following command to run the segmentation. You would need fat-selective (_0000.nii.gz), water-selective (_0001.nii.gz)), and opposed-phase (_0002.nii.gz) MR images
```
nnUNetv2_predict -i /some/input/directory -o /some/output/directory -d 301 -c 3d_fullres -p nnUNetResEncUNetMPlans --continue_prediction
```
If whole-body Dixon MR data (neck to feet) is processed, the model outputs a total of 19 classes. If not part of the input data (e.g., when aquiring only the abdomen), the respective classes should not be included.
| Class number | Meaning |
| ------------ | ------- |
| 1            | Subcutaneous adipose tissue |
| 2            | Visceral adipose tissue |
| 3            | Adipose tissue around the knee |
| 4            | Cardiac adipose tissue |
| 5            | Thigh extensors |
| 6            | Thigh adductors |
| 7            | Glutes (*gluteus maximus*, *gluteus minimus*, *gluteus medius*, *M. piriformis* |
| 8            | Iliopsoas |
| 9            | Back muscles |
| 10           | Aorta |
| 11           | Femur |
| 12           | Hip |
| 13           | Sacrum |
| 14           | Vertebral bodies (Lumbar and thoracic spine) |
| 15           | Kidneys |
| 16           | Liver |
| 17           | Spleen |
| 18           | Pancreas |
| 19           | Calves |

### Quantification
Quantification is based on the functions provided in [quantification_from_segmentation.py](quantification_from_segmentation.py).
You need to load the Nifti files of the (stitched) fat- and water-selective images (and a PDFF map if you acquired multi-echo Dixon). This script uses the example of visceral and subcutaneous adipose tissue, liver (PD)FF and muscle (PD)FF quantification. An outline of a script for automated quantification could look like:
```python
import os

import nibabel as nib
import numpy as np
import pandas as pd

from quantification_from_segmentation import quantify_large_adipose_tissue, quantify_liver, quantify_muscle

# Define columns of a complete quantification
complete_cols = ['VAT/L', 'SAT/L', 'Liver(PD)FF/%', 'BackMuscle(PD)FF/%']

img_seg_directory = '/path/to/converted_image_directory'
pred_seg_directory = '/path/to/predicted_segmentations_directory'
completed_segmentations = [i for i in os.listdir(pred_seg_directory) if i.endswith('.gz')]

# I assume that you store your results in a CSV file
# Of course, any other option is possible (e.g., XLSX,...)
data_csv_file = '/path/to/results/STUDY_ID_MRI_Results.csv'

if os.path.isfile(data_csv_file):
    df = pd.read_csv(data_csv_file, index_col=0)
    if not (df.columns == complete_cols).all():
        df = pd.DataFrame(columns=complete_cols)
else: df = pd.DataFrame(columns=complete_cols)
# Identify segmentations that have not been quantified yet
new_segmentations = [i[:-7] for i in completed_segmentations if i[:-7] not in df.index.values]
results_dict = {}
for new in new_segmentations:
    results = []
    # Load segmentation
    seg = nib.load(os.path.join(pred_seg_directory, f'{new}.nii.gz'))
    in_plane_resolution = np.prod(seg.header['pixdim'][1:4])
    # Load images
    f_img = nib.load(os.path.join(img_seg_directory, f'{new}_0000.nii.gz')).get_fdata()
    w_img = nib.load(os.path.join(img_seg_directory, f'{new}_0001.nii.gz')).get_fdata()
    # Create FF map (or load PDFF map from Nifti file)
    ff_img = np.divide(f_img, (f_img+w_img),
                       out=np.zeros_like(f_img),
                       where=((f_img+w_img) != 0))
    ff_img = ff_img * 1000  # Optional; included here to match SIEMENS PDFF maps
    n_vat, n_sat_abd, n_sat_tho = quantify_large_adipose_tissue(seg)
    results.append(n_vat * in_plane_resolution * 1e-6)
    results.append((n_sat_abd + n_sat_tho) * in_plane_resolution * 1e-6)
    # Liver
    _, ff_liver = quantify_liver(seg, ff_img)
    results.append(ff_liver)
    # Back muscles
    _, _, ff_muscle = quantify_muscle(seg, ff_img, label_muscle=9)
    results.append(ff_muscle)
    # Optional: round results
    results = [np.round(i, 2) if i is not None else i for i in results]
    results_dict[new] = results
# Combine existing df with new results
df = pd.concat([df, pd.DataFrame.from_dict(results_dict, orient='index', columns=complete_cols)])
df.to_csv(data_csv_file)
```
