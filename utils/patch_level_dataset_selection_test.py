import argparse

from utils.patch_level_dataset_selection import PatchLevelDatasetSelection


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches/',
                        help='Source data dir.')
    parser.add_argument('--connected_component_threshold',
                        type=int,
                        default=1,
                        help='The threshold to select legal positive patches.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches-connected-component-1/',
                        help='Destination data dir.')

    args = parser.parse_args()

    return args


def TestPatchLevelDatasetSelection(args):
    obj = PatchLevelDatasetSelection(args.data_root_dir, args.connected_component_threshold, args.output_dir)
    obj.run()

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelDatasetSelection(args)
