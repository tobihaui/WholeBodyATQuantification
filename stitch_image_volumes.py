import os
from typing import List


def stitch_blocks(blocks_to_stitch: List[str],
                  block_directory: str,
                  full_output_file: str):
    if len(blocks_to_stitch) > 1:
        origNodes = []
        for i, b in enumerate(blocks_to_stitch):
            slicer.util.loadVolume(os.path.join(block_directory, b))
            origNodes.append(slicer.util.getNode(f'vtkMRMLScalarVolumeNode{i+1}'))
        roiNode = stitch_logic.createAutomaticROI(origNodes)
        pn.SetNodeReferenceID("StitchedVolumeROI", roiNode.GetID())
        roiNode = pn.GetNodeReference("StitchedVolumeROI")
        voxelSpacingMm = origNodes[0].GetSpacing()
        output_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        output_node = stitch_logic.blend_volumes(origNodes, roiNode,
                                                 voxelSpacingMm=voxelSpacingMm,
                                                 thresholdValue=thresholdValue,
                                                 defaultVoxelValue=defaultVoxelValue,
                                                 useThresholdValue=useThresholdValue)
        pn.SetNodeReferenceID("OutputVolume", output_node.GetID())
        slicer.util.saveNode(output_node, full_output_file)
