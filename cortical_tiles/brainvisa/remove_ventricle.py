#!python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
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

""" Remove ventricle from a volume through the automatic labelled graph
by Morphologist.
/!\ Be careful :
With the BIDS argument, the session-acquisition-run Morphologist folder
should be replaced by a '*' in the <path_to_graph> argument.
"""

import argparse
import glob
import re
import sys
from os.path import abspath, basename, join, exists, isdir
import numpy as np
from p_tqdm import p_map
from soma import aims

from cortical_tiles.brainvisa import exception_handler
from cortical_tiles.brainvisa.utils.folder import create_folder
from cortical_tiles.brainvisa.utils.subjects import get_number_subjects
from cortical_tiles.brainvisa.utils.subjects import select_subjects_int
from cortical_tiles.brainvisa.utils.logs import setup_log
from cortical_tiles.brainvisa.utils.parallel import define_njobs
from cortical_tiles.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_files_general
from cortical_tiles.brainvisa.utils.resample import resample
from cortical_tiles.brainvisa.generate_ICBM2009c_transforms import \
    get_transform_filename
from cortical_tiles.config.logs import set_file_logger

# Import constants
from cortical_tiles.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, \
    _SKELETON_DIR_DEFAULT, _SIDE_DEFAULT, \
    _PATH_TO_GRAPH_DEFAULT

_OUTPUT_DIR_DEFAULT = join(_SKELETON_DIR_DEFAULT, "without_ventricle")
_SRC_FILENAME_DEFAULT = "skeleton_generated_"
_OUTPUT_FILENAME_DEFAULT = "skeleton_generated_without_ventricle_"
_LABELLING_SESSION_DEFAULT = "deepcnn_session_auto"

# Defines logger
log = set_file_logger(__file__)


class VentricleVoxelOutOfBoundsError(IndexError):
    """A ventricle voxel index falls outside the target volume's grid."""


def _check_voxels_in_bounds(voxels, shape, bucket_name):
    """Raise VentricleVoxelOutOfBoundsError if any voxel index is outside shape."""
    if (voxels < 0).any() or (voxels >= shape).any():
        bad = voxels[(voxels < 0).any(axis=1) | (voxels >= shape).any(axis=1)]
        raise VentricleVoxelOutOfBoundsError(
            f"ventricle voxel(s) outside volume grid {tuple(shape)}: "
            f"bucket {bucket_name}, e.g. {tuple(bad[0])}"
        )


def _get_ventricle_buckets(labelled_graph):
    """Returns [(bucket_name, voxels_ndarray), ...] for every non-empty
    ventricle bucket in labelled_graph, across all vertices/edges."""
    buckets = []
    for vertex in labelled_graph.vertices():
        label = vertex.get("label", "unknown")
        if label.startswith("ventricle"):
            for edge in range(len(vertex.edges())):
                for bucket_name in ("aims_plidepassage", 'aims_junction'):
                    if bucket_name in vertex.edges()[edge]:
                        voxels = np.array(
                            vertex.edges()[edge][bucket_name][0].keys())
                        if voxels.shape == (0,):
                            continue
                        buckets.append((bucket_name, voxels))
            for bucket_name in ('aims_bottom', 'aims_other', 'aims_ss'):
                bucket = vertex.get(bucket_name)
                if bucket is not None:
                    voxels = np.array(bucket[0].keys())
                    if voxels.shape == (0,):
                        continue
                    buckets.append((bucket_name, voxels))
    return buckets


def _make_ventricle_marker(voxels, voxel_size):
    """Builds an S16 aims.Volume with 1 at each voxel index, 0 elsewhere,
    sized to fit every voxel. Raises VentricleVoxelOutOfBoundsError on a
    negative index (marker sizing can't accommodate that)."""
    if (voxels < 0).any():
        bad = voxels[(voxels < 0).any(axis=1)]
        raise VentricleVoxelOutOfBoundsError(
            f"ventricle voxel(s) have negative index: e.g. {tuple(bad[0])}"
        )
    dims = list((voxels.max(axis=0) + 1).astype(int))
    marker = aims.Volume(dims, dtype="S16")
    marker.header()['voxel_size'] = list(voxel_size)
    marker.fill(0)
    for i, j, k in voxels:
        marker[i, j, k] = 1
    return marker


def _compute_resampled_ventricle_mask(labelled_graph, transform, output_vs):
    """Returns a boolean ndarray (on the resampled grid) of voxels the
    ventricle maps onto, or None if this graph has no ventricle voxels."""
    buckets = _get_ventricle_buckets(labelled_graph)
    if not buckets:
        return None
    voxel_size = labelled_graph["voxel_size"]
    mask = None
    for _, voxels in buckets:
        marker = _make_ventricle_marker(voxels, voxel_size)
        resampled = resample(marker, transform, output_vs=output_vs)
        side_mask = np.asarray(resampled)[..., 0] != 0
        mask = side_mask if mask is None else (mask | side_mask)
    return mask


def remove_ventricle_from_graph(volume, labelled_graph, background=0,
                                 transform=None):
    arr = np.asarray(volume)
    shape = arr.shape[:3]

    if transform is not None:
        output_vs = np.asarray(volume.header()['voxel_size'][:3])
        mask = _compute_resampled_ventricle_mask(
            labelled_graph, transform, output_vs)
        if mask is None:
            return volume
        if mask.shape != shape:
            raise VentricleVoxelOutOfBoundsError(
                f"resampled ventricle mask shape {mask.shape} does not "
                f"match target volume grid {shape}"
            )
        arr[mask] = background
        return volume

    for bucket_name, voxels in _get_ventricle_buckets(labelled_graph):
        _check_voxels_in_bounds(voxels, shape, bucket_name)
        for i, j, k in voxels:
            arr[i, j, k] = background
    return volume


def parse_args(argv):
    """Parses command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description="Generate volumes without ventricle.")
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SKELETON_DIR_DEFAULT,
        help="Directory where are the files you want ventricle "
             "to be removed from."
             f"Default is : {_SKELETON_DIR_DEFAULT}")
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_OUTPUT_DIR_DEFAULT,
        help="Output directory where to put the files without ventricle. "
             f"Default is : {_OUTPUT_DIR_DEFAULT}")
    parser.add_argument(
        "-m", "--morpho_dir", type=str, default=_SRC_DIR_DEFAULT,
        help="Directory where the graph data lies. It has to point directly to"
             "the morphologist directory containing the subjects "
             "as subdirectories. "
             f"Default is : {_SRC_DIR_DEFAULT}")
    parser.add_argument(
        "-p", "--path_to_graph", type=str,
        default=_PATH_TO_GRAPH_DEFAULT,
        help="Relative path to graph. "
        "In BIDS format, the session_acquisition_run directory "
        "have to be replaced by *. "
        f"Default is :  {_PATH_TO_GRAPH_DEFAULT}")
    parser.add_argument(
        "-l", "--labelling_session",
        default=_LABELLING_SESSION_DEFAULT,
        help="Name of the labelling session in Morphologist tree. "
             f"Default is : {_LABELLING_SESSION_DEFAULT}")
    parser.add_argument(
        "-f", "--src_filename", type=str,
        default=_SRC_FILENAME_DEFAULT,
        help="Filename of source files. "
        "Format is : <SIDE><source_filename>_<SUBJECT>.nii.gz "
        f"Default is : {_SRC_FILENAME_DEFAULT}")
    parser.add_argument(
        "-e", "--output_filename", type=str, default=_OUTPUT_FILENAME_DEFAULT,
        help="Filename of output files. "
             "Format is : <SIDE><output_filename>_<SUBJECT>.nii.gz "
             f"Default is : {_OUTPUT_FILENAME_DEFAULT}")
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help="Hemisphere side (either L, R or F)"
             f"Default is : {_SIDE_DEFAULT}")
    parser.add_argument(
        "-b", "--bids", default=False, action="store_true",
        help="if the database uses BIDS format"
             "Default is : False")
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')

    args = parser.parse_args(argv)
    dico_suffix = {"R": "right", "L": "left", "F": "full"}
    setup_log(args,
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix=dico_suffix[args.side])

    params = vars(args)

    params['output_dir'] = abspath(args.output_dir)
    params['morpho_dir'] = abspath(args.morpho_dir)
    params['src_dir'] = abspath(args.src_dir)
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


def _get_side_list(side):
    return ["L", "R"] if side == "F" else [side]


def _get_subject_id(subject):
    if subject.startswith("_"):
        return subject[1:]
    return subject


class RemoveVentricleFromVolume:
    """Class to remove ventricle from volume files through the automatic
    labelling graph computed by Morphologist. The default automatic labelling
    session is : deepcnn_session_auto.

    It contains all the information to get labelled graphs from volume
    filenames and to write new volumes without ventricle in the target
    directory.
    """

    def __init__(self, src_dir, output_dir,
                 morpho_dir, path_to_graph, labelling_session,
                 src_filename, output_filename,
                 side, bids, parallel, transform_dir=None):

        self.side = side
        self.bids = bids
        self.parallel = parallel
        self.morpho_dir = morpho_dir
        self.path_to_graph = path_to_graph
        self.labelling_session = labelling_session
        self.transform_dir = transform_dir
        self.src_dir = join(src_dir, self.side)
        self.output_dir = join(output_dir, self.side)
        create_folder(abspath(self.output_dir))

        self.expr = f"{self.side}{src_filename}(.*).nii.gz$"
        self.src_filename = src_filename
        self.output_filename = output_filename
        self.src_file = f"%(side)s{src_filename}%(subject)s.nii.gz"
        self.output_file = f"%(side)s{output_filename}%(subject)s.nii.gz"

    def remove_ventricle_from_one_subject(self, subject: str):
        """ Removes ventricle and writes new volume file for one subject.
        """

        sbj = {"subject": subject, "side": self.side}

        src_file = join(self.src_dir, self.src_file % sbj)

        output_file = join(self.output_dir, self.output_file % sbj)

        labelled_graph_list = self.get_labelled_graph(subject)

        log.debug(f"src_file = {src_file}")
        log.debug(f"labelled_graphs = {labelled_graph_list}")
        log.debug(f"output_file = {output_file}")

        side_list = _get_side_list(self.side)

        try:
            if exists(src_file):
                volume = aims.read(src_file)
                for side, graph_file in zip(side_list, labelled_graph_list):
                    if exists(graph_file):
                        labelled_graph = aims.read(graph_file)
                        transform = None
                        if self.transform_dir is not None:
                            transform = self.get_transform_file(
                                subject, side, graph_file)
                        volume = remove_ventricle_from_graph(
                            volume, labelled_graph, transform=transform)
                    else:
                        raise FileNotFoundError(f"Labelled graph not found : \
                                                {graph_file}")
                aims.write(volume, output_file)
            else:
                raise FileNotFoundError(f"Source file not found : \
                                        {src_file}")
        except VentricleVoxelOutOfBoundsError as e:
            log.error(f"{subject}: {repr(e)}")
            raise VentricleVoxelOutOfBoundsError(f"subject {subject}: {e}") from e
        except Exception as e:
            log.error(f"{subject}: {repr(e)}")

    def get_transform_file(self, subject, side, graph_file):
        """Returns the .trm path for one subject/side, matching what
        generate_ICBM2009c_transforms wrote at self.transform_dir."""
        subject_id = _get_subject_id(subject)
        side_transform_dir = join(self.transform_dir, side)
        return get_transform_filename(
            side_transform_dir, side, subject_id, graph_file, self.bids)

    def get_labelled_graph(self, subject: str):
        """ Find the labelled graph in the morphologist database from the
        source filename.
        """
        labelled_graph_list = []
        side_list = _get_side_list(self.side)
        subject = _get_subject_id(subject)
        for side in side_list:
            if self.bids:
                split = subject.split("_")
                subject_id = split[0]
                if len(split) > 1:
                    keys = "_".join(split[1:])
                else:
                    keys = ""
                    log.warning(f"The subject {subject} has no session, "
                                "acquisition or run.")
                filename = f"{side}{subject_id}_{self.labelling_session}.arg"
                labelled_graph_file = join(
                    self.morpho_dir, subject_id, self.path_to_graph.replace(
                        "*", keys), self.labelling_session, filename)
            else:
                filename = f"{side}{subject}_{self.labelling_session}.arg"
                labelled_graph_file = join(
                    self.morpho_dir,
                    subject,
                    self.path_to_graph,
                    self.labelling_session,
                    filename)
            labelled_graph_list.append(labelled_graph_file)
        return labelled_graph_list

    def compute(self, number_subjects):
        """Loops over subjects and remove ventricle from volumes.
        """
        # Gets list of subjects
        log.debug(f"src_dir = {self.src_dir}")
        log.debug(f"reg exp = {self.expr}")

        if isdir(self.src_dir):
            src_files = glob.glob(f"{self.src_dir}/*.nii.gz")
            log.debug(f"Volume files list = {src_files}")

            # Generates list of subjects not treated yet
            not_processed_files = get_not_processed_files_general(
                self.src_dir, self.output_dir,
                self.src_filename, self.output_filename)

            list_not_processed_subjects = [
                re.search(self.expr, basename(dI))[1]
                for dI in not_processed_files]
            list_all_subjects = [
                re.search(self.expr, basename(dI))[1]
                for dI in src_files]

            list_subjects = select_subjects_int(
                list_all_subjects,
                list_not_processed_subjects,
                number_subjects)

            log.info(f"Expected number of subjects = {len(list_subjects)}")
            log.info(f"list_subjects[:5] = {list_subjects[:5]}")
            log.debug(f"list_subjects = {list_subjects}")
        else:
            raise NotADirectoryError(
                f"{self.src_dir} doesn't exist or is not a directory")

        # Performs computation on all subjects either serially or in parallel
        if self.parallel:
            log.info(
                "PARALLEL MODE: subjects are computed in parallel.")
            p_map(self.remove_ventricle_from_one_subject,
                  list_subjects,
                  num_cpus=define_njobs())
        else:
            log.info(
                "SERIAL MODE: subjects are scanned serially, "
                "without parallelism")
            for sub in list_subjects:
                self.remove_ventricle_from_one_subject(sub)

        # Checks if there is expected number of generated files
        compare_number_aims_files_with_expected(self.output_dir, list_subjects)


def remove_ventricle(src_dir=_SRC_DIR_DEFAULT,
                     output_dir=_OUTPUT_DIR_DEFAULT,
                     morpho_dir=_SRC_DIR_DEFAULT,
                     path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                     labelling_session=_LABELLING_SESSION_DEFAULT,
                     src_filename=_SRC_FILENAME_DEFAULT,
                     output_filename=_OUTPUT_FILENAME_DEFAULT,
                     side=_SIDE_DEFAULT,
                     bids=False,
                     parallel=False,
                     number_subjects=_ALL_SUBJECTS,
                     transform_dir=None):
    """Remove ventricle from a volume
    through the automatic labelled graph by Morphologist"""

    # Initialization
    removal = RemoveVentricleFromVolume(
        src_dir=src_dir,
        output_dir=output_dir,
        morpho_dir=morpho_dir,
        path_to_graph=path_to_graph,
        labelling_session=labelling_session,
        src_filename=src_filename,
        output_filename=output_filename,
        side=side,
        bids=bids,
        parallel=parallel,
        transform_dir=transform_dir)
    removal.compute(number_subjects=number_subjects)


@exception_handler
def main(argv):
    """Reads argument line and remove ventricle from volumes
    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    remove_ventricle(
        src_dir=params["src_dir"],
        output_dir=params["output_dir"],
        morpho_dir=params["morpho_dir"],
        path_to_graph=params["path_to_graph"],
        labelling_session=params["labelling_session"],
        src_filename=params["src_filename"],
        output_filename=params["output_filename"],
        side=params["side"],
        bids=params["bids"],
        parallel=params['parallel'],
        number_subjects=params['nb_subjects'])


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
