# ruff: noqa: E501 - keys are exact call texts printed by the rule test
"""Exemptions from the listing-order rule of cortical_tiles (TASK-READORDER-BDRABCZUK-4550903D1CB2).

tests/test_listing_order_rule.py fails on a glob, iglob, rglob, listdir, scandir, walk or iterdir
call of the cortical_tiles package that is not inside a sorted() call, unless the call is listed
here with the reason its order never reaches a result (a set, a membership or emptiness test, a
count, ...).

Key: (module path relative to the cortical_tiles repository root, qualified name of the enclosing
function or class or "<module>", ast.unparse() text of the call). The rule test prints the keys of
the calls it reports in this form. Value: the reason, a non-empty text.
"""

EXEMPTIONS: dict[tuple[str, str, str], str] = {
    ('cortical_tiles/brainvisa/benchmark_generation_distmap.py', 'get_sub_list', 'os.listdir(_DEFAULT_DATA_DIR)'): 'membership test only; the subject list order comes from subjects_list',
    ('cortical_tiles/brainvisa/generate_ICBM2009c_transforms.py', 'GraphGenerateTransform.compute', "glob.glob(f'{self.src_dir}/*/{self.path_to_graph}')"): 'only the number of files is compared',
    ('cortical_tiles/brainvisa/generate_skeletons.py', 'GraphConvert2Skeleton.compute', "glob.glob(f'{self.src_dir}/*/{self.path_to_graph}')"): 'only the number of files is compared',
    ('cortical_tiles/brainvisa/generate_distmaps.py', 'SkelConvert2DistMap.find_skeleton_file', "glob.glob(f'{glob.escape(search_dir)}/*{suffix}')"): 'filtered into matches, which are sorted and must hold exactly one file',
    ('cortical_tiles/brainvisa/generate_distmaps.py', 'SkelConvert2DistMap.find_skeleton_file', "glob.glob(f'{glob.escape(search_dir)}/' + glob.escape(name))"): 'exact name; matches are sorted and must hold exactly one file',
    ('cortical_tiles/brainvisa/generate_one_sulcal_region.py', 'is_directory_empty', 'os.listdir(path)'): 'emptiness test only',
    ('cortical_tiles/brainvisa/generate_one_sulcal_region.py', '_resampled_subjects', 'os.listdir(side_dir)'): 'collected into a set',
    ('cortical_tiles/brainvisa/generate_sulcal_regions.py', '_keeps_a_session', 'glob.glob(f"{config[\'graphs_dir\']}/{subject}/{config[\'path_to_graph\']}/{side}*.arg")'): 'collected into a set',
    ('cortical_tiles/brainvisa/utils/crop_subjects.py', '_subjects_in', 'os.listdir(folder)'): 'collected into a set',
    ('cortical_tiles/brainvisa/utils/mask_qc.py', 'check', 'os.listdir(morpho_dir)'): 'random.sample result is overwritten by a fixed subject list on the next line',
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'compare_number_aims_files_with_expected', "glob.glob(f'{output_dir}/*')"): 'only the number of files is compared',
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'compare_number_aims_files_with_number_in_source', "glob.glob(f'{output_dir}/*')"): 'only the number of files is compared',
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'compare_number_aims_files_with_number_in_source', "glob.glob(f'{src_dir}/*')"): 'only the number of files is compared',
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_files', "glob.glob(f'{tgt_dir}/*.nii.gz')"): "set difference of subjects; the order reaches only a 'first file' log line",
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_files_general', "glob.glob(f'{tgt_dir}/*.nii.gz')"): "set difference of subjects; the order reaches only a 'first file' log line",
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_subjects_dict', "glob.glob(f'{tgt_dir}/*.nii.gz')"): "set difference of subjects; the order reaches only a 'first file' log line",
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_cropped_files', "glob.glob(f'{tgt_dir}/*.nii.gz')"): "set difference of subjects; the order reaches only a 'first file' log line",
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_subjects', "glob.glob(f'{tgt_dir}/*.nii.gz')"): "set difference of subjects; the order reaches only a 'first file' log line",
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_subjects_distmap', "glob.glob(f'{tgt_dir}/*.nii.gz')"): "set difference of subjects; the order reaches only a 'first file' log line",
    ('cortical_tiles/brainvisa/utils/quality_checks.py', 'get_not_processed_subjects_transform', "glob.glob(f'{tgt_dir}/*.trm')"): "set difference of subjects; the order reaches only a 'first file' log line",
}
