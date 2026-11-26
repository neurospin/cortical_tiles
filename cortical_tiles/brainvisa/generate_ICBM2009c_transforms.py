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

"""Write transform files to ICBM 2009c from graph files

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python generate_ICBM2009c_transforms.py


"""

import glob
import sys
import re
from os.path import basename
from os.path import isdir

from cortical_tiles.brainvisa import exception_handler
from cortical_tiles.brainvisa import graph_step
from cortical_tiles.brainvisa.subject_graphs import stop_on_several_labelling_sets
from cortical_tiles.brainvisa.utils.subjects import \
    select_subjects_int, select_good_qc
from cortical_tiles.brainvisa.utils.parallel import define_njobs
from cortical_tiles.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_subjects_transform
from cortical_tiles.brainvisa.utils.skip_list import \
    covered_by, load_skipped, session_key
from pqdm.processes import pqdm
from cortical_tiles.config.logs import set_file_logger
from soma import aims

# Import constants
from cortical_tiles.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, \
    _TRANSFORM_DIR_DEFAULT, _SIDE_DEFAULT, \
    _QC_PATH_DEFAULT

# Defines logger
log = set_file_logger(__file__)

# Member of the skip-list step set (skip_list.SKIP_STEPS)
_STEP = "generate_ICBM2009c_transforms"


_CLI = graph_step.StepCli(prog=basename(__file__),
                          description='Generates transformation files ' 'to ICBM 2009c template from graphs',
                          output_key='transform_dir', output_default=_TRANSFORM_DIR_DEFAULT,
                          output_help='Output directory where to put skeleton files.' 'Default is : ' + _TRANSFORM_DIR_DEFAULT,
                          bids_help="If the database uses the BIDS format", junction=False)


def parse_args(argv):
    """Parses command-line arguments"""
    return graph_step.parse_step_args(argv, _CLI)


def get_transform_filename(side_transform_dir, side, subject, graph_file, bids):
    """Builds the .trm filename for one subject/side's ICBM2009c transform.

    side_transform_dir is the side-specific directory
    (i.e. <transform_dir>/<side>), matching GraphGenerateTransform's own
    self.transform_dir layout.
    """
    key = session_key(subject, graph_file, bids)
    return f"{side_transform_dir}/{side}transform_to_ICBM2009c_{key}.trm"


class GraphGenerateTransform(graph_step.GraphStep):
    """Class to write the ICBM2009c transforms of all graphs from a folder"""

    STEP = _STEP  # noqa: V107 - used through graph_step.GraphStep
    LOG = log  # noqa: V107 - used through graph_step.GraphStep
    ONE_SUBJECT = "generate_one_transform"  # noqa: V107 - used through graph_step.GraphStep
    PARALLEL_MESSAGE = "PARALLEL MODE: transforms are generated in parallel."  # noqa: V107 - used through graph_step.GraphStep
    SERIAL_MESSAGE = "SERIAL MODE: transforms are generated serially, without parallelism"  # noqa: V107 - used through graph_step.GraphStep

    def __init__(self, src_dir, transform_dir,
                 side, parallel,
                 path_to_graph, bids, qc_path, skip_dir=None, graph_label=None, graphs=None):
        super().__init__(src_dir, transform_dir, side, parallel, path_to_graph, bids, qc_path, skip_dir,
                         graph_label, graphs)

    def generate_one_transform(self, subject: str):  # noqa: V105 - used through graph_step.GraphStep
        """Generates and writes ICBM2009c transform for one subject."""
        list_graph_file, graph_path = self.graph_files(self._subject_folders(subject), f"{self.src_dir}/{subject}*")
        log.debug(graph_path)
        log.debug(f"list_graph_file = {list_graph_file}")
        if not list_graph_file:
            return (subject, self.record_missing(subject, f"No graph file: {graph_path} doesn't exist"))
        return (subject, self.convert_graphs(subject, list_graph_file))

    def _subject_folders(self, subject):
        """Returns the src_dir folders of subject, sorted: its own and its session folders."""
        result = [f for f in sorted(glob.glob(f"{self.src_dir}/{subject}*"))
                  if isdir(f) and covered_by(basename(f), {subject})]
        assert result == sorted(result), "GraphReadingSteps._subject_folders post: result == sorted(result)"
        return result

    def get_transform_filename(self, subject, graph_file):
        return get_transform_filename(
            self.side_dir, self.side, subject, graph_file, self.bids)

    get_output_filename = get_transform_filename  # noqa: V107 - used through graph_step.GraphStep

    def convert(self, graph_file, output_file):  # noqa: V105 - used through graph_step.GraphStep
        graph = aims.read(graph_file)
        g_to_icbm_template = \
            aims.GraphManip.getICBM2009cTemplateTransform(graph)
        aims.write(g_to_icbm_template, output_file)

    def map_parallel(self, one_subject, list_subjects):  # noqa: V105 - used through graph_step.GraphStep
        return pqdm(list_subjects, one_subject, n_jobs=define_njobs())

    def compute(self, nb_subjects):  # noqa: V105 - used through graph_step.GraphStep
        """Loops over subjects to generate transforms to ICBM2009c from graphs.
        """
        list_subjects = self.listed_subjects()
        log.info(f"Number of subjects before qc = {len(list_subjects)}")
        self.prescan_listed(list_subjects)
        list_subjects = select_good_qc(list_subjects, self.qc_path)
        skipped = load_skipped(self.skip_dir, self.side)
        self.skipped_keys = frozenset(skipped)  # noqa: V101 - used through graph_step.GraphStep
        list_subjects = [s for s in list_subjects if s not in skipped]
        not_processed_subjects = \
            get_not_processed_subjects_transform(list_subjects,
                                                 self.side_dir,
                                                 prefix="ICBM2009c_")
        list_subjects = select_subjects_int(list_subjects,
                                            not_processed_subjects,
                                            nb_subjects)

        log.info(f"Expected number of subjects = {len(list_subjects)}")
        log.info(f"list_subjects[:5] = {list_subjects[:5]}")
        log.debug(f"list_subjects = {list_subjects}")

        results = self.run_subjects(list_subjects)
        self.log_summary("Transform generation", results, list_subjects)

        # Checks if there is expected number of generated files
        if self.counts_listed_graphs():
            list_graphs = [g for g in
                           glob.glob(f"{self.src_dir}/*/{self.path_to_graph}")
                           if not re.search('.minf$', g)]
            compare_number_aims_files_with_expected(
                self.side_dir, list_graphs)
        else:
            compare_number_aims_files_with_expected(self.side_dir,
                                                    list_subjects)


def generate_ICBM2009c_transforms(src_dir=_SRC_DIR_DEFAULT, transform_dir=_TRANSFORM_DIR_DEFAULT, path_to_graph=None,
                                  side=_SIDE_DEFAULT, bids=False, parallel=False, nb_subjects=_ALL_SUBJECTS,
                                  qc_path=_QC_PATH_DEFAULT, skip_dir=None, graph_label=None, graphs=None):
    """Generates skeletons from graphs

    skip_dir: directory of the per-side skip lists (cortical_tiles skip_list
    module). Default: transform_dir.
    """
    GraphGenerateTransform.run(locals())


@exception_handler
@stop_on_several_labelling_sets
def main(argv):
    """Reads argument line and generates transform to ICBM2009c from graph

    Args:
        argv: a list containing command line arguments
    """
    generate_ICBM2009c_transforms(**parse_args(argv))

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
