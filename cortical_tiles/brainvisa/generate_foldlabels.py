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

"""Write foldlabels from graph files

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python generate_foldlabels.py


"""

import sys
from os.path import basename

from p_tqdm import p_map

from cortical_tiles.brainvisa import exception_handler
from cortical_tiles.brainvisa import graph_step
from cortical_tiles.brainvisa.subject_graphs import stop_on_several_labelling_sets
from cortical_tiles.brainvisa.utils.subjects import select_subjects_int, \
    select_good_qc
from cortical_tiles.brainvisa.utils.parallel import define_njobs
from cortical_tiles.brainvisa.utils.foldlabel import \
    generate_foldlabel_from_graph_file
from cortical_tiles.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_subjects, \
    save_list_to_csv
from cortical_tiles.brainvisa.utils.skip_list import \
    load_skipped, session_key
from cortical_tiles.config.logs import set_file_logger

# Import constants
from cortical_tiles.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, \
    _FOLDLABEL_DIR_DEFAULT, _SIDE_DEFAULT, \
    _JUNCTION_DEFAULT, \
    _QC_PATH_DEFAULT

# Defines logger
log = set_file_logger(__file__)

# Member of SKIP_STEPS under which this step records its failures
_STEP = "generate_foldlabels"


_CLI = graph_step.StepCli(prog=basename(__file__), description='Generates foldlabel files from graphs',
                          output_key='foldlabel_dir', output_default=_FOLDLABEL_DIR_DEFAULT,
                          output_help='Output directory where to put fold label files.' 'Default is : ' + _FOLDLABEL_DIR_DEFAULT,
                          src_help='Source directory where the graph data lies.' 'Default is : ' + _SRC_DIR_DEFAULT,
                          side_help='Hemisphere side (either L or R).', skip_dir=True)


def parse_args(argv):
    """Parses command-line arguments"""
    return graph_step.parse_step_args(argv, _CLI)


class GraphConvert2FoldLabel(graph_step.GraphStep):
    """Class to convert graph into foldlabel files"""

    STEP = _STEP  # noqa: V107 - used through graph_step.GraphStep
    LOG = log  # noqa: V107 - used through graph_step.GraphStep
    ONE_SUBJECT = "generate_one_foldlabel"  # noqa: V107 - used through graph_step.GraphStep

    def __init__(self, src_dir, foldlabel_dir,
                 side, junction, parallel,
                 path_to_graph, bids, qc_path, skip_dir=None, graph_label=None, graphs=None):
        self.junction = junction
        super().__init__(src_dir, foldlabel_dir, side, parallel, path_to_graph, bids, qc_path, skip_dir,
                         graph_label, graphs)

    def get_foldlabel_filename(self, subject, graph_file):
        """Build foldlabel file name"""
        key = session_key(subject, graph_file, self.bids)
        return f"{self.side_dir}/{self.side}foldlabel_{key}.nii.gz"

    get_output_filename = get_foldlabel_filename  # noqa: V107 - used through graph_step.GraphStep

    def convert(self, graph_file, output_file):  # noqa: V105 - used through graph_step.GraphStep
        generate_foldlabel_from_graph_file(graph_file, output_file, self.junction)

    def map_parallel(self, one_subject, list_subjects):  # noqa: V105 - used through graph_step.GraphStep
        return p_map(one_subject, list_subjects, num_cpus=define_njobs())

    def generate_one_foldlabel(self, subject: str):  # noqa: V105 - used through graph_step.GraphStep
        """Generates and writes foldlabel for one subject."""
        # Gets graph file path
        list_graph_file, graph_path = self.graph_files([f"{self.src_dir}/{subject}"], f"{self.src_dir}/{subject}")
        log.debug(f"list_graph_file = {list_graph_file}")
        if not list_graph_file:
            return subject, self.record_missing(subject, f"No graph file: {graph_path} doesn't exist")
        return subject, self.convert_graphs(subject, list_graph_file)

    def compute(self, nb_subjects):  # noqa: V105 - used through graph_step.GraphStep
        """Loops over subjects and converts graphs into skeletons.
        """
        list_subjects = self.listed_subjects()
        self.prescan_listed(list_subjects)
        log.info(f"Number of subjects before qc = {len(list_subjects)}")
        qc_subjects = select_good_qc(list_subjects, self.qc_path)
        skipped = load_skipped(self.skip_dir, self.side)
        self.skipped_keys = frozenset(skipped)  # noqa: V101 - used through graph_step.GraphStep
        processable = [s for s in qc_subjects if s not in skipped]
        not_processed_subjects = get_not_processed_subjects(
            processable, self.side_dir, "foldlabel_")

        list_subjects = select_subjects_int(processable,
                                            not_processed_subjects,
                                            nb_subjects)
        results = self.run_subjects(list_subjects)
        self.log_summary("Foldlabel generation", results, list_subjects)

        # Checks if there is the expected number of generated files
        compare_number_aims_files_with_expected(self.side_dir,
                                                list_subjects)
        not_processed_subjects = get_not_processed_subjects(
            qc_subjects, self.side_dir, "foldlabel_")
        save_list_to_csv(
            not_processed_subjects,
            f"{self.side_dir}/../not_processed_files_{self.side}.csv")


def generate_foldlabels(src_dir=_SRC_DIR_DEFAULT, foldlabel_dir=_FOLDLABEL_DIR_DEFAULT, path_to_graph=None,
                        side=_SIDE_DEFAULT, junction=_JUNCTION_DEFAULT, bids=False, parallel=False,
                        nb_subjects=_ALL_SUBJECTS, qc_path=_QC_PATH_DEFAULT, skip_dir=None, graph_label=None,
                        graphs=None):
    """Generates foldlabels from graphs"""
    GraphConvert2FoldLabel.run(locals())


@exception_handler
@stop_on_several_labelling_sets
def main(argv):
    """Reads argument line and generates foldlabel from graph

    Args:
        argv: a list containing command line arguments
    """
    generate_foldlabels(**parse_args(argv))


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
