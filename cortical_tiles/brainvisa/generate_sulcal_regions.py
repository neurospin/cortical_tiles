"""
Uses the pipeline on multiple regions and datasets,
for both sides and input type."""

import argparse
import glob
import os
from os.path import basename, join
import json
import sys

from cortical_tiles.brainvisa.utils.crop_subjects import \
    NoValidSubjectError, exclude_empty_crops, mirror_empty_crops
from cortical_tiles.brainvisa.utils.folder import get_nth_parent_dir
from cortical_tiles.brainvisa.utils.skip_list import \
    clear_skip_list, load_skip_entries, load_skipped
from cortical_tiles.brainvisa.utils.subjects import \
    is_it_a_subject, select_good_qc

from cortical_tiles.brainvisa import exception_handler
from cortical_tiles.config.logs import set_file_logger
from generate_one_sulcal_region import crop_region_dir, run_with_params

# Defines logger
log = set_file_logger(__file__)

# The relative path leads to right outside of cortical_tiles directory and /data/ which is the prefered file architecture for accessing the data
#_PATH_DATASET_ROOT_DEFAULT = os.path.join(get_nth_parent_dir(os.getcwd(), 3), 'data/') #"/neurospin/dico/data/cortical_tiles/current/datasets"
# _DATASETS_DEFAULT = ["UkBioBank40"]
_CORTICAL_TILES_VERSION = "2026"
# Exit codes of the script; 2 is argparse's
_EXIT_NO_VALID_SUBJECT = 1
_EXIT_PARTIAL = 3
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
        help='Path where cortical_tiles dataset lie.',
        required=True)
    parser.add_argument(
        "-o", "--output_dir", type=str,
        help="Path where cortical_tiles derivatives will lie." \
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
    parser.add_argument(
        "--masks", type=str, default=None,
        help="Mask version tag (e.g. 'canonical_25'). "
             "Overrides masks_version in the pipeline JSON config."
    )
    parser.add_argument(
        "--skel_seed", type=int, default=None,
        help="VipSkeleton seed for the 2mm re-skeletonization. "
             "Overrides skel_seed in the pipeline JSON config."
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
        """Runs each side and input type; returns the sides that raised NoValidSubjectError"""
        raised_sides = set()
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
                try:
                    run_with_params(cfg)
                except NoValidSubjectError as exc:
                    log.warning("Region %s side %s input %s: %s",
                                cfg["region_name"], side, input_type, exc)
                    raised_sides.add(side)
                    continue
                print(
                    f"\nEND\n"
                    f"{cfg['region_name']} {side} {input_type} ok\n"
                )
        return raised_sides


# ($local config key, argument that must replace it)
_LOCAL_GRAPH_ARGUMENTS = (("path_to_graph", "path_to_graph"),
                          ("path_to_skeleton_with_hull", "path_sk_with_hull"))


def _check_local_graph_arguments(config: dict, arguments: dict) -> None:
    """Raises ValueError if a $local graph path has no argument to replace it"""
    for key, argument in _LOCAL_GRAPH_ARGUMENTS:
        if config.get(key) == "$local" and not arguments[argument]:
            raise ValueError(
                f"{key} is '$local' in pipeline_loop_2mm.json but argument "
                f"{argument} is empty; pass --{argument} (no default: "
                "Morphologist graph layouts differ)")


def _run_regions(resolved_config, regions, sides, input_types, njobs):
    """Runs every region; returns the sides that raised NoValidSubjectError

    Under save_behavior clear_and_compute, the skip list of each side is
    emptied once, before the first region. A subject with an empty skeleton
    crop reaches the skip list of each side after each region, and leaves the
    stacks and cropped files of each region and side after the last region.
    """
    output_dir = resolved_config["output_dir"]
    if resolved_config.get("save_behavior") == "clear_and_compute":
        for side in sides:
            clear_skip_list(output_dir, side)

    raised_sides = set()
    dropped = set()
    for region in regions:
        raised_sides |= RegionPipelineRunner(resolved_config, region).run(
            sides, input_types, njobs
        )
        dropped = mirror_empty_crops(output_dir, sides)
    if dropped:
        exclude_empty_crops(
            [crop_region_dir(resolved_config, region) for region in regions],
            dropped, sides)
    return raised_sides


def generate_sulcal_regions(regions, sides, input_types,
                            path_dataset, verbose, output_dir, path_to_graph,
                            path_sk_with_hull, sk_qc_path, njobs, masks=None,
                            skel_seed=None):
    """Global loops to generate all regions for all dataset

    Returns:
        (resolved_config, raised_sides), raised_sides being the set of the
        sides whose run_with_params call raised NoValidSubjectError
    """

    # Load and resolve the template ONCE — never written back
    pipeline_json = f"{path_dataset}/pipeline_loop_2mm.json"
    with open(pipeline_json, 'r') as f:
        resolved_config = json.load(f)

    if masks:
        resolved_config["masks_version"] = masks
    if skel_seed is not None:
        resolved_config["skel_seed"] = skel_seed

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
        _check_local_graph_arguments(
            resolved_config,
            {"path_to_graph": path_to_graph,
             "path_sk_with_hull": path_sk_with_hull})
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
                    output_dir if output_dir not in ("", None)
                    else f"derivatives/cortical_tiles-{_CORTICAL_TILES_VERSION}"
                )
            elif k == "path_to_graph":
                resolved_config[k] = path_to_graph
            elif k == "path_to_skeleton_with_hull":
                resolved_config[k] = path_sk_with_hull
            elif k == "skel_qc_path":
                resolved_config[k] = sk_qc_path

    return resolved_config, _run_regions(
        resolved_config, regions, sides, input_types, njobs)


def valid_subjects(config, side):
    """Returns the subjects of graphs_dir kept by QC and absent from the skip list of side"""
    names = [basename(path)
             for path in glob.glob(f"{config['graphs_dir']}/*")
             if is_it_a_subject(path)]
    kept = select_good_qc(names, config.get("skel_qc_path", ""))
    return set(kept) - load_skipped(config["output_dir"], side)


def _log_side_summary(side, side_entries, clause):
    """Logs one ERROR record per step holding an entry in side_entries"""
    by_step = {}
    for entry in side_entries:
        by_step.setdefault(entry.step, []).append(entry)
    for step, step_entries in by_step.items():
        subjects = sorted({entry.subject for entry in step_entries})
        causes = dict.fromkeys(f"{entry.exc_type}: {entry.message}"
                               for entry in step_entries)
        log.error("Run summary: side %s, step %s: %d skipped subject(s): %s; "
                  "causes: %s%s", side, step, len(subjects),
                  ", ".join(subjects), " | ".join(causes), clause)


def log_run_summary(entries, valid, raised_sides):
    """Logs the skipped subjects of each side and step, and the sides lost with no entry

    Args:
        entries: dict side -> list of SkipEntry, keyed by the sides of the run
        valid: dict side -> set of the valid subjects of that side
        raised_sides: set of the sides that raised NoValidSubjectError
    """
    for side, side_entries in entries.items():
        if side_entries:
            clause = ""
            if not valid[side] and any(valid.values()):
                clause = f"; side {side} keeps no valid subject"
            _log_side_summary(side, side_entries, clause)
        elif side in raised_sides:
            log.error("Run summary: side %s lost: NoValidSubjectError raised, "
                      "no skip entry", side)


def run_exit_code(entries, valid, raised_sides):
    """Returns the exit code of a run that finished the region loop

    Args:
        entries: dict side -> list of SkipEntry, keyed by the sides of the run
        valid: dict side -> set of the valid subjects of that side
        raised_sides: set of the sides that raised NoValidSubjectError

    Returns:
        1 when no side keeps a valid subject, else 3 when a subject was
        skipped or a side raised, else 0
    """
    if not any(valid.values()):
        return _EXIT_NO_VALID_SUBJECT
    if raised_sides or any(entries.values()):
        return _EXIT_PARTIAL
    return 0


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
    outcome = generate_sulcal_regions(**params)
    if outcome is None:
        return
    config, raised_sides = outcome
    sides = params["sides"]
    entries = {side: load_skip_entries(config["output_dir"], side)
               for side in sides}
    valid = {side: valid_subjects(config, side) for side in sides}
    log_run_summary(entries, valid, raised_sides)
    code = run_exit_code(entries, valid, raised_sides)
    if code:
        sys.exit(code)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
