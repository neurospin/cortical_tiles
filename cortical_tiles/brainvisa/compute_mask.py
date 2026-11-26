#!python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
"""
The aim of this script is to output bounding box and mask for a given sulcus
based on a manually labelled dataset

Bounding box corresponds to the biggest box that encompasses the given sulci
on all subjects of the manually labelled dataset. It measures the bounding box
in the MNI152  space.
"""
import os
import sys
from os.path import basename
from os.path import join

import numpy as np
from cortical_tiles.brainvisa import exception_handler
from cortical_tiles.brainvisa import subject_cli
from cortical_tiles.brainvisa.subject_graphs import stop_on_several_labelling_sets
from cortical_tiles.brainvisa.utils.folder import create_folder
from cortical_tiles.brainvisa.utils.referentials import \
    generate_ref_volume_ICBM2009c
from cortical_tiles.brainvisa.utils.subjects import \
    select_subjects_int_if_list_of_dict
from cortical_tiles.brainvisa.utils.subjects import \
    get_all_subjects_as_dictionary
from cortical_tiles.brainvisa.utils.quality_checks import \
    get_not_processed_subjects_dict
from cortical_tiles.config.logs import set_file_logger
from soma import aims

# Imports constants
from cortical_tiles.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SUPERVISED_SRC_DIR_DEFAULT, \
    _MASK_DIR_DEFAULT, _SIDE_DEFAULT, _SULCUS_DEFAULT, \
    _VOXEL_SIZE_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def initialize_mask(out_voxel_size: tuple) -> aims.Volume:
    """Creates aims volume in MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip

    Args:
        output_voxel_size: tuple
            Output voxel size (default: None, no resampling)

    Returns:
        mask: volume (aims.Volume_S16) filled with 0 in MNI2009 referential
            and with requested voxel_size
    """

    return generate_ref_volume_ICBM2009c(out_voxel_size)


def increment_one_mask(graph_filename, mask, sulcus, voxel_size_out):
    """Increments self.mask of 1 where there is the sulcus

    Args:
        graph_filename: string being the name of graph file .arg to analyze:
                        for example: 'Lammon_base2018_manual.arg'

    """

    # Reads the data graph and transforms it to MNI ICBM152 referential
    graph = aims.read(graph_filename)
    g_to_icbm_template = aims.GraphManip.getICBM2009cTemplateTransform(
        graph)
    voxel_size_in = graph['voxel_size'][:3]
    arr = np.asarray(mask)
    arr_one = np.zeros(arr.shape).astype(np.int16)

    # Gets the min and max coordinates of the sulci
    # by looping over all the vertices of the graph
    for vertex in graph.vertices():
        vname = vertex.get('name')
        if vname != sulcus:
            continue
        for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels_real = np.asarray(
                    [g_to_icbm_template.transform(
                        np.array(voxel) * voxel_size_in)
                        for voxel in bucket[0].keys()])
                if voxels_real.shape == (0,):
                    continue
                voxels = np.round(np.array(voxels_real) /
                                  voxel_size_out[:3]).astype(int)

                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_one[i, j, k, 0] += 1

    # Guarantees that array is one 1 even if vosel is present twice
    arr_one = (arr_one >= 1).astype(np.int16)

    # Puts in aims volume the array of one mask
    vol_one = aims.Volume(arr_one)
    vol_one.copyHeaderFrom(mask.header())
    vol_one.header()['voxel_size'] = mask.header()['voxel_size']

    # Increments global mask
    arr += arr_one

    return vol_one


def increment_mask(subjects: list,
                   mask: aims.Volume,
                   sulcus: str,
                   voxel_size_out: tuple,
                   sample_dir: str, graph_spec=None):
    """Increments mask for the chosen sulcus for all subjects

    Parameters:
        subjects: list containing all subjects to be analyzed
        graph_spec: graph specification of graph discovery,
            or None for the subjects' legacy graph_file glob
    """

    for sub, graph_path in subject_cli.iter_subject_graphs(subjects, log, graph_spec):
        vol_one = increment_one_mask(graph_path, mask, sulcus, voxel_size_out)
        aims.write(vol_one, f"{sample_dir}/{sub['subject']}.nii.gz")


def write_mask(mask: aims.Volume, mask_file: str):
    """Writes mask on mask file"""
    mask_file_dir = os.path.dirname(mask_file)
    os.makedirs(mask_file_dir, exist_ok=True)
    log.info(f"\nFinal mask file: {mask_file}\n")
    aims.write(mask, mask_file)


class MaskAroundSulcus(subject_cli.SulcusOverSubjects):
    """Determines the mask around given sulci

    It is determined in the  MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip
    """

    def __init__(self, src_dir=_SUPERVISED_SRC_DIR_DEFAULT, path_to_graph=None,
                 mask_dir=_MASK_DIR_DEFAULT, sulcus=_SULCUS_DEFAULT, new_sulcus=None, side=_SIDE_DEFAULT,
                 out_voxel_size=None, graph_label=None):
        """Inits with list of directories and list of sulci

        Attributes:
            src_dir: list of strings naming full path source directories
            path_to_graph: string naming relative path to labelled graph
                (None: graph discovery below each subject folder)
            mask_dir: name of target directory with full path
            sulcus: sulcus name
            new_sulcus: new sulcus name
            side: hemisphere side (either L for left,
                                or R for right hemisphere)
            out_voxel_size: float for voxel size in mm
            graph_label: labelling set read during graph discovery
        """

        super().__init__(src_dir, path_to_graph, sulcus, new_sulcus, side, graph_label)
        self.mask_dir = mask_dir
        self.mask_sample_dir = f"{mask_dir}/{self.side}/{self.new_sulcus}"
        self.out_voxel_size = (out_voxel_size,
                               out_voxel_size,
                               out_voxel_size)

        # Initiliazes mask
        self.mask = aims.Volume()
        self.mask_file = join(
            self.mask_dir,
            self.side,
            self.new_sulcus + '.nii.gz')

    def compute(self, number_subjects=_ALL_SUBJECTS):
        """Main class program to compute the bounding box

        Args:
            number_subjects: number_subjects to analyze
        """

        if number_subjects:
            subjects = get_all_subjects_as_dictionary(
                self.src_dir,
                self.graph_file,
                self.side)
            # Stops on several labelling sets before any folder or graph read
            subject_cli.prescan_subjects(subjects, self.graph_spec)

            # Creates target mask_dir if they don't exist
            create_folder(self.mask_dir)
            create_folder(self.mask_sample_dir)

            # Generates list of subjects not treated yet
            not_treated_subjects = get_not_processed_subjects_dict(
                subjects, self.mask_sample_dir)

            if len(not_treated_subjects):
                # Gives the possibility to list only the first number_subjects
                subjects = select_subjects_int_if_list_of_dict(subjects,
                                               not_treated_subjects,
                                               number_subjects)

                # Creates volume that will take the mask
                self.mask = initialize_mask(self.out_voxel_size)

                # Increments mask for each sulcus and subjects
                increment_mask(subjects,
                               self.mask,
                               self.sulcus,
                               self.out_voxel_size,
                               self.mask_sample_dir, self.graph_spec)

                log.info(f"{self.sulcus}: max = {self.mask.np.max()}")
                # Saving of generated masks
                write_mask(self.mask, self.mask_file)


def compute_mask(src_dir=_SUPERVISED_SRC_DIR_DEFAULT, mask_dir=_MASK_DIR_DEFAULT,
                 path_to_graph=None, sulcus=_SULCUS_DEFAULT, new_sulcus=None,
                 side=_SIDE_DEFAULT, number_subjects=_ALL_SUBJECTS, out_voxel_size=None, graph_label=None):
    """ Main program computing the box encompassing the sulcus in all subjects

    The programm loops over all subjects
    and computes in MNI space the voxel coordinates of the box encompassing
    the sulci for all subjects

    Args:
        src_dir: list of strings -> directories of the supervised databases
        mask_dir: string giving target mask directory path
        path_to_graph: string giving relative path to manually labelled graph
            (None: graph discovery below each subject folder)
        side: hemisphere side (either 'L' for left, or 'R' for right)
        sulcus: string giving the sulcus to analyze
        new_sulcus: string giving a new sulcus name (optional)
        number_subjects: integer giving the number of subjects to analyze,
            by default it is set to _ALL_SUBJECTS (-1)
        out_voxel_size: float giving voxel size
        graph_label: labelling set read during graph discovery

    Returns:
        aims volume containing the mask
    """
    mask = MaskAroundSulcus(src_dir=src_dir, mask_dir=mask_dir, path_to_graph=path_to_graph, sulcus=sulcus,
                            new_sulcus=new_sulcus, side=side, out_voxel_size=out_voxel_size, graph_label=graph_label)
    mask.compute(number_subjects=number_subjects)

    return mask.mask


# Command-line texts of this script
_CLI = subject_cli.ScriptCli(
    prog=basename(__file__), description='Computes mask around the named sulcus', output_key='mask_dir',
    output_default=_MASK_DIR_DEFAULT,
    output_help='Output directory where to store the output mask files. Default is : ' + _MASK_DIR_DEFAULT,
    sulcus_help='Sulcus name around which we determine the bounding box/mask. Default is : ' + _SULCUS_DEFAULT,
    voxel_size_help='Voxel size of mask. Default is : ' + str(_VOXEL_SIZE_DEFAULT))


def parse_args(argv: list) -> dict:
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: a dictionary with all arugments as keys
    """

    return subject_cli.parse_subject_args(argv, _CLI)


@exception_handler
@stop_on_several_labelling_sets
def main(argv):
    """Reads argument line and determines the max bounding box

    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)
    # Actual API
    compute_mask(
        src_dir=params['src_dir'],
        path_to_graph=params['path_to_graph'],
        mask_dir=params['mask_dir'],
        sulcus=params['sulcus'],
        new_sulcus=params['new_sulcus'],
        side=params['side'],
        number_subjects=params['nb_subjects'],
        out_voxel_size=params['out_voxel_size'], graph_label=params['graph_label'])


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
