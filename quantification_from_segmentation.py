"""Collection of functions for adipose tissue and ectopic fat quantification
from segmentations using whole-body Dixon MRI.

Import these functions in a postprocessing script loading segmentations (for 
adipose tissue volume quantification) and PDFF/FF% maps (for ectopic fat 
quantification).
                                                         
Most functions can be parametrized and have only been used with suggested 
default parameters -- without thereby excluding other possible values as more
suitable for individual contexts.
A rationale for each parameter suggestions will be found in the respective
function descriptions.
"""
import cc3d
import nibabel as nib
import numpy as np

from typing import Tuple, Union, Dict
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.ndimage import center_of_mass


# VOLUME QUANTIFICATION
## Subcutaneous and visceral adipose tissue
def quantify_large_adipose_tissue(pred_: nib.Nifti1Image,
                                  label_sat: int = 1, label_vat: int = 2,
                                  label_thigh_ext: int = None,
                                  label_thigh_add: int = None) -> Union[Tuple[int,
                                                                              int,
                                                                              int,
                                                                              int,
                                                                              int],
                                                                        Tuple[int,
                                                                              int,
                                                                              int]]:
    pred_ = pred_.get_fdata()
    # Iterate from head to feet
    # Avoid the head - sometimes, there are missegmentations.
    for i in range(pred_.shape[-1]-51, -1, -1):
        if (pred_[:, :, i] == label_vat).any() \
            and (pred_[:, :, i] == label_sat).any():
            sat_sep_top = i
            break
        else:
            sat_sep_top = None
    for i in range(pred_.shape[-1]):
        if (pred_[:, :, i] == label_vat).any():
            sat_sep_bottom = i
            break
        else:
            sat_sep_bottom = None
    n_vat = np.count_nonzero(pred_[:, :, sat_sep_bottom:sat_sep_top] == label_vat)
    n_sat_abd = np.count_nonzero(pred_[:, :, sat_sep_bottom:sat_sep_top] == label_sat)
    n_sat_tho = np.count_nonzero(pred_[:, :, sat_sep_top:] == label_sat)
    
    if label_thigh_add and label_thigh_ext is not None:
        for i in range(pred_.shape[-1]):
            if (pred_[:, :, i] == label_thigh_add).any() \
                and (pred_[:, :, i] == label_thigh_ext).any():
                gfat_sep = i
                break
            else:
                gfat_sep = None
        n_gfat = np.count_nonzero(pred_[:, :, gfat_sep:sat_sep_bottom] == label_sat)
        n_calves = np.count_nonzero(pred_[:, :, :gfat_sep] == label_sat)
    
        return n_vat, n_sat_abd, n_sat_tho, n_gfat, n_calves
    else:
        return n_vat, n_sat_abd, n_sat_tho


## Cardiac adipose tissue
def quantify_cardiac_adipose_tissue(pred_: nib.Nifti1Image,
                                    label_card_at: int = 4) -> int:
    pred_ = pred_.get_fdata()
    n_card_at = np.count_nonzero(pred_[pred_ == label_card_at])
    
    return n_card_at


# ECTOPIC FAT QUANTIFICATION
## Renal hilum fat
"""
Macroscopic fat inside the kidneys
"""
def quantify_renal_hilum_fat(pred_: nib.Nifti1Image,
                             ff_map_: np.array,
                             label_kidneys: int = 15,
                             save_modified_mask: str = None) -> Tuple[int, int]:
    pred_data_ = pred_.get_fdata()
    fat_in_hilum = np.zeros(pred_data_.shape, dtype=np.int8)
    kidney_seg = np.where(pred_data_ == label_kidneys, label_kidneys, 0)
    kidney_seg = erode_volume(kidney_seg)
    sep = left_right_separation(kidney_seg)
    kidney_seg[:sep, :, :][kidney_seg[:sep, :, :] != 0] = 20
    kidney_seg[sep:, :, :][kidney_seg[sep:, :, :] != 0] = 21
    thr_l = compute_intensity_threshold(ff_map_[kidney_seg == 20])
    thr_r = compute_intensity_threshold(ff_map_[kidney_seg == 21])
    fat_in_hilum[(kidney_seg == 20) & (ff_map_ > thr_l)] = 1
    fat_in_hilum[(kidney_seg == 21) & (ff_map_ > thr_r)] = 2
    
    n_hilum_l = np.count_nonzero(fat_in_hilum == 1)
    n_hilum_r = np.count_nonzero(fat_in_hilum == 2)
    
    if save_modified_mask is not None:
        _save_modified_mask(fat_in_hilum, pred_, save_modified_mask)
    
    return n_hilum_l, n_hilum_r


## Intramuscular fat
"""
Macroscopic fat between muscle groups
(PD)FF in remaining tissue
"""
def quantify_muscle(pred_: nib.Nifti1Image,
                    ff_map_: np.array,
                    label_muscle: int = 5,
                    two_sided: bool = False,
                    save_modified_mask: str = None) -> Tuple[int, int, float]:
    
    pred_data_ = pred_.get_fdata()
    imat = np.zeros(pred_data_.shape, dtype=np.int8)
    muscle_seg = np.where(pred_data_ == label_muscle, label_muscle, 0)
    n_muscle = np.count_nonzero(muscle_seg)
    muscle_seg = erode_volume(muscle_seg)
    thr = compute_intensity_threshold(ff_map_[muscle_seg != 0])
    if two_sided:
        print('Not implemented yet.')
    else:
        imat[(muscle_seg != 0) & (ff_map_ > thr)] = label_muscle
        lean_muscle = muscle_seg - imat
        n_imat = np.count_nonzero(imat == label_muscle)
        
        ff_lean = np.mean(ff_map_[lean_muscle != 0]) * 0.1
        
        if save_modified_mask is not None:
            _save_modified_mask(imat, pred_, save_modified_mask)
        
        return n_muscle, n_imat, ff_lean


## Liver
def quantify_liver(pred_: nib.Nifti1Image,
                   ff_map_: np.array,
                   label_liver: int = 16,
                   save_modified_mask: str = None) -> Tuple[int, float]:
    pred_data_ = pred_.get_fdata()
    liver_seg = np.where(pred_data_ == label_liver, label_liver, 0)
    n_liver = np.count_nonzero(liver_seg)
    
    # Perform erosion for (PD)FF quantification
    liver_seg = erode_volume(liver_seg)
    ff_liver = np.mean(ff_map_[liver_seg != 0]) * 0.1
    
    return n_liver, ff_liver


## Intrapancreatic fat
"""
Volume of macroscopic fat with 1-50 with 3x3 erosion threshold
(PD)FF in remaining tissue
"""
def quantify_pancreas(pred_: nib.Nifti1Image,
                      ff_map_: np.array,
                      label_panc: int = 18,
                      save_modified_mask: str = None) -> Tuple[int, int, float]:
    
    pred_data_ = pred_.get_fdata()
    panc_seg = np.where(pred_data_ == label_panc, label_panc, 0)
    n_panc = np.count_nonzero(panc_seg)
    
    # Perform erosion for (PD)FF quantification
    panc_seg = erode_volume(panc_seg)
    ipat = np.zeros(pred_data_.shape, dtype=np.int8)
    ipat[(panc_seg != 0) & (ff_map_ > 500)] = 1
    n_ipat = np.count_nonzero(ipat)
    
    lean_panc = panc_seg - ipat
    ff_panc = np.mean(ff_map_[lean_panc != 0 & (ff_map_ >= 10)]) * 0.1
    
    if save_modified_mask is not None:
        _save_modified_mask(ipat, pred_, save_modified_mask)
    
    return n_panc, n_ipat, ff_panc


## Vertebral bodies
def quantify_vertebral_fat_fraction(pred_: nib.Nifti1Image,
                                    ff_map_: np.array,
                                    label_verteb: int = 14,
                                    save_modified_mask: str = None) -> Tuple[Dict[str,
                                                                                  float],
                                                                             Dict[str,
                                                                                  float]]:
    v_names = ['L5', 'L4', 'L3', 'L2', 'L1', 'Th12', 'Th11', 'Th10', 'Th9',
               'Th8', 'Th7', 'Th6', 'Th5', 'Th4', 'Th3', 'Th2', 'Th1']
    pred_data_ = pred_.get_fdata()
    verteb_seg = np.where(pred_data_ == label_verteb, label_verteb, 0)
    verteb_seg = erode_volume(verteb_seg)
    # Connected components labeling of vertebral bodies
    vertebral_body = cc3d.connected_components(verteb_seg)
    cc3d_new_labels = [i for i in np.unique(vertebral_body) if i != 0]
    vert_body_stats = cc3d.statistics(vertebral_body)
    
    vert_body_ff = {}
    vert_body_vol = {}
    if len(cc3d_new_labels) < len(v_names):
        for l, n, v in zip(cc3d_new_labels,
                           vert_body_stats['voxel_counts'][1:],
                           v_names[:len(cc3d_new_labels)]):
            vert_body_ff[v] = np.mean(ff_map_[vertebral_body == l]) * 0.1
            vert_body_vol[v] = n
    else:
        for l, n, v in zip(cc3d_new_labels[:len(v_names)],
                           vert_body_stats['voxel_counts'][1:len(v_names)+1],
                           v_names):
            vert_body_ff[v] = np.mean(ff_map_[vertebral_body == l]) * 0.1
            vert_body_vol[v] = n    
    
    return vert_body_ff, vert_body_vol


## Other bones?
def quantify_bone_marrow(pred_: nib.Nifti1Image,
                         ff_map_: np.array,
                         label_bones: int = 11,
                         two_sided: bool = False,
                         save_modified_mask: str = None):

    pred_data_ = pred_.get_fdata()
    bone_seg = np.where(pred_data_ == label_bones, 1, 0)
    
    # Perform erosion for (PD)FF quantification
    bone_seg = erode_volume(bone_seg)
    ff_bone = np.mean(ff_map_[bone_seg != 0]) * 0.1

    return ff_bone


# periX ADIPOSE TISSUE QUANTIFICATION
def quantify_perix_adipose_tissue(pred_: nib.Nifti1Image,
                                  ff_map_: np.array,
                                  peri_x_label: int,
                                  dilation_structure: np.array = np.ones((1, 5, 5), np.int8),
                                  ff_intensity_threshold: int = 500):
    
    pred_data_ = pred_.get_fdata()
    base_seg = np.where(pred_data_ == peri_x_label, 1, 0)
    dilated_seg = dilate_volume(base_seg)
    diff = dilated_seg - base_seg
    n_peri_x = np.count_nonzero(ff_map_[(diff != 0) & (ff_map_ > 500)])
    
    return n_peri_x


# Helper functions
def compute_intensity_threshold(ff_map_: np.array) -> float:
    try:
        q3 = np.quantile(ff_map_, 0.75)
        iqr = q3 - np.quantile(ff_map_, 0.25)
        thr = q3 + 3 * iqr
        # Clip threshold of high fat patients
        # Value is empirically determined
        if thr > 200: thr = 200
    except:
        thr = 200

    return thr


def left_right_separation(pred_one_class_: np.array) -> int:
    # Center of mass in left/right
    com_class = int(center_of_mass(pred_one_class_)[0])  
    left_right_sep = 0
    if pred_one_class_[com_class, :, :].any():
        for i in range(com_class, com_class + 20, 1):
            if not pred_one_class_[i, :, :].any():
                left_right_sep = i
                break
        if left_right_sep == 0:
            for i in range(com_class, com_class - 20, -1):
                if not pred_one_class_[i, :, :].any():
                    left_right_sep = i
                    break
    else:
        left_right_sep = com_class
        
    return left_right_sep


def erode_volume(pred_: np.array,
                 kernel: np.array = np.ones((1, 3, 3), np.int8)) -> np.array:
    """

    Parameters
    ----------
    pred_ : np.array
        Segmentation mask.
    kernel : np.array, optional
        Size of the structuring element for binary erosion. The default is
        np.ones((1, 3, 3), np.int8).

    Returns
    -------
    pred_ : np.array
        The eroded segmentation mask.

    """
    pred_ = binary_erosion(pred_, kernel).astype(pred_.dtype)
    
    return pred_


def dilate_volume(pred_: np.array,
                  kernel: np.array = np.ones((1, 5, 5), np.int8)) -> np.array:
    """
    
    Parameters
    ----------
    pred_ : np.array
        Segmentation mask.
    kernel : np.array, optional
        Size of the structuring element for binary dilation. The default is
        np.ones((1, 5, 5), np.int8) which results in a two-voxel "ring" around
        the target structure.

    Returns
    -------
    pred_ : np.array
        The dilated segmentation mask.

    """
    pred_ = binary_dilation(pred_, kernel).astype(pred_.dtype)
    
    return pred_


def _save_modified_mask(pred_modified_: np.array,
                        pred_: nib.Nifti1Image,
                        full_path_and_filename: str):
    modified_mask = nib.Nifti1Image(pred_modified_, pred_.affine, pred_.header)
    nib.save(modified_mask, full_path_and_filename)
    
