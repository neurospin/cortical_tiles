"""
Uses the pipeline on multiple regions and datasets,
for both sides and input type."""

import argparse
import os
from os.path import join
import json
import sys

from deep_folding.brainvisa.utils.folder import get_nth_parent_dir

from deep_folding.brainvisa import exception_handler
from deep_folding.config.logs import set_file_logger
from generate_one_sulcal_region import run_with_params

# Defines logger
log = set_file_logger(__file__)

# The relative path leads to right outside of deep_folding directory and /data/ which is the prefered file architecture for accessing the data
#_PATH_DATASET_ROOT_DEFAULT = os.path.join(get_nth_parent_dir(os.getcwd(), 3), 'data/') #"/neurospin/dico/data/deep_folding/current/datasets"
# _DATASETS_DEFAULT = ["UkBioBank40"]
_CORTICAL_TILES_VERSION = "2026"
_SIDES_DEFAULT = ["L", "R"]
_INPUT_TYPES_DEFAULT = ["skeleton", "foldlabel", "extremities"]
_REGIONS_DEFAULT = ["S.C.-sylv.", "S.C.-S.Pe.C.", "S.C.-S.Po.C.",\
            "S.Pe.C.", "S.Po.C.", "S.F.int.-F.C.M.ant.",\
            "S.F.inf.-BROCA-S.Pe.C.inf.", "S.T.s.", "Sc.Cal.-S.Li.",\
            "F.C.M.post.-S.p.C.", "S.T.i.-S.O.T.lat.",\
            "OCCIPITAL", "F.I.P.-F.I.P.Po.C.inf.", "S.F.inter.-S.F.sup.",\
            "S.F.median-S.F.pol.tr.-S.F.sup.", "S.Or.",\
            "S.Or.-S.Olf.", "F.P.O.-S.Cu.-Sc.Cal.",\
            "S.s.P.-S.Pa.int.", "S.T.s.br.",\
            "Lobule_parietal_sup.", "S.F.marginal-S.F.inf.ant.",\
            "F.Coll.-S.Rh.", "S.T.i.-S.T.s.-S.T.pol.",\
            "F.C.L.p.-subsc.-F.C.L.a.-INSULA.", "S.F.int.-S.R.",\
            "S.Call.", "S.Call.-S.s.P.-S.intraCing."\
            ]


def parse_args(argv):
    """Function parsing command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Generates all specified sulcal regions')
    parser.add_argument(
        "-d", "--path_dataset", type=str,
        help='Path where deep_folding dataset lie.',
        required=True)
    parser.add_argument(
        "-o", "--output_dir", type=str,
        help="Path where deep_folding derivatives will lie." \
        "Default is $DATASET_PATH/derivatives/."
    )
    parser.add_argument(
        "--path_to_graph", type=str, required=True
    )
    parser.add_argument(
        "--path_sk_with_hull", type=str, required=True
    )
    parser.add_argument(
        "--sk_qc_path", type=str, default=""
    )
    # parser.add_argument(
    #     "-d", "--datasets", type=str, default=_DATASETS_DEFAULT, nargs='+',
    #     help='Datasets to process. '
    #          'Give all desired datasets one after the other. '
    #          'Example: -d dataset1 dataset2'
    #          'Default is : ' + ' '.join(_DATASETS_DEFAULT))
    parser.add_argument(
        "-i", "--sides", type=str, default=_SIDES_DEFAULT, nargs='+',
        help='Hemisphere side (either L or R). '
             'Gives the desired sides one after the other. '
             'Example: -i L R'
             'Default is : ' + ' '.join(_SIDES_DEFAULT))
    parser.add_argument(
        "-y", "--input_types", type=str, default=_INPUT_TYPES_DEFAULT, nargs='+',
        help='Input types: \'skeleton\', \'foldlabel\', \'extremities\'. '
        'Give the desired types one after the other. '
        'Example: -y skeleton foldlabel'
        'Default is : ' + ' '.join(_INPUT_TYPES_DEFAULT))
    parser.add_argument(
        "-r", "--regions", type=str, default=_REGIONS_DEFAULT, nargs='+',
        help='Give desired sulcal regions. '
             'Gives the desired sulcal regions one after the other. '
             'Example: -r S.C.-sylv. S.F.inf.-BROCA-S.Pe.C.inf.'
             'Default is : ' + ' '.join(_REGIONS_DEFAULT))
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')
    parser.add_argument(
        "--njobs", help="Number of CPU cores allowed to use. Default is your maximum number of cores - 2 or up to 22 if you have enough cores.",
        type=int
    )

    params = {}

    args = parser.parse_args(argv)

    params = vars(args)

    verbose = '-' + ('v' * args.verbose) if args.verbose > 0 else ''
    

    # params['crop_dir'] = args.output_dir
    # params['list_sulci'] = args.sulcus  # a list of sulci

    # # Checks if nb_subjects is either the string "all" or a positive integer
    # params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    # # Removes renamed params
    # # So that we can use params dictionary directly as function arguments
    # params.pop('output_dir')
    # params.pop('sulcus')
    params.pop('verbose')
    params['verbose'] = verbose

    return params


class RegionPipelineRunner:
    """Runs the full pipeline for one region, entirely in-memory.

    Instantiate with the already-resolved config dict and the region name.
    Call run() to iterate over all sides and input_types.
    No files are written — run_with_params() is called directly in-process.
    """

    def __init__(self, resolved_config: dict, region: str):
        self.config = dict(resolved_config)
        self.config["region_name"] = region
        self.config["combine_type"] = (region == "CINGULATE.")
        self.config["threshold"] = (
            1 if region in (
                "OCCIPITAL", "F.C.L.p.-subsc.-F.C.L.a.-INSULA.")
            else 0
        )

    def run(self, sides, input_types, njobs):
        for side in sides:
            for input_type in input_types:
                cfg = dict(self.config)
                cfg["side"] = side
                cfg["input_type"] = input_type
                cfg["njobs"] = njobs
                # Side-specific threshold override
                if (cfg["region_name"] ==
                        "F.C.L.p.-subsc.-F.C.L.a.-INSULA."
                        and side == "L"):
                    cfg["threshold"] = 1
                run_with_params(cfg)
                print(
                    f"\nEND\n"
                    f"{cfg['region_name']} {side} {input_type} ok\n"
                )


def generate_sulcal_regions(regions, sides, input_types,
                            path_dataset, verbose, output_dir, path_to_graph,
                            path_sk_with_hull, sk_qc_path, njobs):
    """Global loops to generate all regions for all dataset"""

    # Load and resolve the template ONCE — never written back
    pipeline_json = f"{path_dataset}/pipeline_loop_2mm.json"
    with open(pipeline_json, 'r') as f:
        resolved_config = json.load(f)

    if "$local" not in resolved_config.values():
        for src, key in {
            path_to_graph: "path_to_graph",
            path_sk_with_hull: "path_to_skeleton_with_hull"
        }.items():
            if src:
                resolved_config[key] = src
        # Always overwrite skel_qc_path (even when empty) to prevent
        # stale values from a previous run persisting in the config
        resolved_config["skel_qc_path"] = sk_qc_path
    else:
        for k, v in list(resolved_config.items()):
            if v != "$local":
                continue
            if k == "brain_regions_json":
                resolved_config[k] = join(
                    get_nth_parent_dir(os.getcwd(), 4),
                    'sulci_regions_champollion_V1.json'
                )
            elif k == "supervised_output_dir":
                resolved_config[k] = join(
                    get_nth_parent_dir(os.getcwd(), 3),
                    'cortical_tiles/data'
                )
            elif k == "graphs_dir":
                resolved_config[k] = join(
                    path_dataset, "derivatives/morphologist-6.0"
                )
            elif k == "output_dir":
                resolved_config[k] = join(
                    path_dataset,
                    f"derivatives/cortical_tiles-{_CORTICAL_TILES_VERSION}"
                    if output_dir not in ("", None) else output_dir
                )
            elif k == "path_to_graph" and path_to_graph:
                resolved_config[k] = path_to_graph
            elif k == "path_to_skeleton_with_hull" and path_sk_with_hull:
                resolved_config[k] = path_sk_with_hull
            elif k == "skel_qc_path":
                resolved_config[k] = sk_qc_path

    for region in regions:
        RegionPipelineRunner(resolved_config, region).run(
            sides, input_types, njobs
        )


@exception_handler
def main(argv):
    """Reads argument line and generates sulcal regions
    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)
    print(params)

    # Actual API
    generate_sulcal_regions(**params)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
