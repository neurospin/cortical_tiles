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

"""Write skeletons from graph files

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python generate_skeletons.py


"""

import glob
import re
import sys
from os.path import basename

from p_tqdm import p_map

from cortical_tiles.brainvisa import exception_handler
from cortical_tiles.brainvisa import graph_step
from cortical_tiles.brainvisa.subject_graphs import stop_on_several_labelling_sets
from cortical_tiles.brainvisa.utils.subjects import \
    select_subjects_int, select_good_qc
from cortical_tiles.brainvisa.utils.parallel import define_njobs
from cortical_tiles.brainvisa.utils.skip_list import \
    load_skipped, session_key
from cortical_tiles.brainvisa.utils.skeleton import \
    generate_skeleton_from_graph_file
from cortical_tiles.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_subjects
from cortical_tiles.config.logs import set_file_logger

# Import constants
from cortical_tiles.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, \
    _SKELETON_DIR_DEFAULT, _SIDE_DEFAULT, \
    _JUNCTION_DEFAULT, \
    _QC_PATH_DEFAULT, \
    _NB_JOBS_DEFAULT

# Defines logger
log = set_file_logger(__file__)

# Step name of this module in the skip list (one of SKIP_STEPS)
_STEP = "generate_skeletons"


_CLI = graph_step.StepCli(prog=basename(__file__), description='Generates skeleton files from graphs',
                          output_key='skeleton_dir', output_default=_SKELETON_DIR_DEFAULT,
                          output_help='Output directory where to put skeleton files.' 'Default is : ' + _SKELETON_DIR_DEFAULT)


def parse_args(argv):
    """Parses command-line arguments"""
    return graph_step.parse_step_args(argv, _CLI)


class GraphConvert2Skeleton(graph_step.GraphStep):
    """Class to convert all graphs from a folder into skeletons"""

    STEP = _STEP  # noqa: V107 - used through graph_step.GraphStep
    LOG = log  # noqa: V107 - used through graph_step.GraphStep
    ONE_SUBJECT = "generate_one_skeleton"  # noqa: V107 - used through graph_step.GraphStep

    def __init__(self, src_dir, skeleton_dir,
                 side, junction, parallel,
                 path_to_graph, bids, qc_path, njobs, skip_dir=None, graph_label=None, graphs=None):
        self.junction = junction
        self.njobs = njobs
        super().__init__(src_dir, skeleton_dir, side, parallel, path_to_graph, bids, qc_path, skip_dir,
                         graph_label, graphs)

    def get_skeleton_filename(self, subject, graph_file):
        key = session_key(subject, graph_file, self.bids)
        return (f"{self.side_dir}/"
                f"{self.side}skeleton_generated_{key}.nii.gz")

    get_output_filename = get_skeleton_filename  # noqa: V107 - used through graph_step.GraphStep

    def convert(self, graph_file, output_file):  # noqa: V105 - used through graph_step.GraphStep
        generate_skeleton_from_graph_file(graph_file, output_file, self.junction)

    def map_parallel(self, one_subject, list_subjects):  # noqa: V105 - used through graph_step.GraphStep
        return p_map(one_subject, list_subjects, num_cpus=define_njobs(self.njobs))

    def generate_one_skeleton(self, subject: str):  # noqa: V105 - used through graph_step.GraphStep
        """Generates and writes skeleton for one subject."""
        list_graph_file, graph_path = self.graph_files([f"{self.src_dir}/{subject}"], f"{self.src_dir}/{subject}")
        log.debug(f"list_graph_file = {list_graph_file}")
        if len(list_graph_file) == 0:
            self.record_missing(subject, f"No graph file for subject {subject} (expected: {graph_path})")
            return (subject, graph_path)
        return None if self.convert_graphs(subject, list_graph_file) is None else (subject, None)

    def compute(self, nb_subjects):  # noqa: V105 - used through graph_step.GraphStep
        """Loops over subjects and converts graphs into skeletons.
        """
        skipped_before = load_skipped(self.skip_dir, self.side)
        self.skipped_keys = frozenset(skipped_before)  # noqa: V101 - used through graph_step.GraphStep
        # Gets list of subjects
        list_subjects = self.listed_subjects()
        self.prescan_listed(list_subjects)
        log.info(f"Number of subjects before qc = {len(list_subjects)}")
        list_subjects = select_good_qc(list_subjects, self.qc_path)
        list_subjects = [s for s in list_subjects if s not in skipped_before]
        not_processed_subjects = \
            get_not_processed_subjects(list_subjects, self.side_dir)

        list_subjects = select_subjects_int(list_subjects,
                                            not_processed_subjects,
                                            nb_subjects)

        log.info(f"Expected number of subjects = {len(list_subjects)}")
        log.info(f"list_subjects[:5] = {list_subjects[:5]}")
        log.debug(f"list_subjects = {list_subjects}")

        results = self.run_subjects(list_subjects)
        failed = [r for r in results if r is not None]
        skipped = [r for r in failed if r[1] is not None]
        n_total = len(list_subjects)
        n_skipped = len(failed)
        n_ok = n_total - n_skipped

        # Always print a per-side summary to stdout
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"SKELETON GENERATION SUMMARY  (side={self.side})")
        print(f"  Total subjects:  {n_total}")
        print(f"  Successful:      {n_ok}")
        print(f"  Skipped:         {n_skipped}")
        if skipped:
            print(f"\n  Skipped subjects (no graph file found):")
            for subject, expected_path in skipped:
                print(f"    {subject}  →  {expected_path}")

            # Write QC CSV so the caller can review / feed back as --sk_qc_path
            import csv
            qc_csv = f"{self.side_dir}/skipped_subjects.csv"
            with open(qc_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["subject", "expected_graph_path"])
                writer.writerows(skipped)
            print(f"\n  QC file written: {qc_csv}")
            print(f"  You can pass it with --sk_qc_path to exclude these"
                  f" subjects from future runs.")
        print(f"{sep}\n")

        # Checks if there is expected number of generated files
        if self.counts_listed_graphs():
            list_graphs = \
                [g for g in glob.glob(f"{self.src_dir}/*/{self.path_to_graph}")
                 if not re.search('.minf$', g)]
            compare_number_aims_files_with_expected(self.side_dir,
                                                    list_graphs)
        else:
            compare_number_aims_files_with_expected(self.side_dir,
                                                    list_subjects)


def generate_skeletons(src_dir=_SRC_DIR_DEFAULT, skeleton_dir=_SKELETON_DIR_DEFAULT, path_to_graph=None,
                       side=_SIDE_DEFAULT, junction=_JUNCTION_DEFAULT, bids=False, parallel=False,
                       nb_subjects=_ALL_SUBJECTS, qc_path=_QC_PATH_DEFAULT, njobs=_NB_JOBS_DEFAULT, skip_dir=None,
                       graph_label=None, graphs=None):
    """Generates skeletons from graphs"""
    GraphConvert2Skeleton.run(locals())


@exception_handler
@stop_on_several_labelling_sets
def main(argv):
    """Reads argument line and generates skeleton from graph

    Args:
        argv: a list containing command line arguments
    """
    generate_skeletons(**parse_args(argv))


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
