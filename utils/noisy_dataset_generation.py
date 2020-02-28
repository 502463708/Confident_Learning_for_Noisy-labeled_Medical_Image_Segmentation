import os
import shutil


def get_sub_dataset_filename_list(src_patch_level_dataset_type_dir, src_radiograph_level_dataset_type_dir, patch_type,
                                  dataset_type, sub_dataset_idx, logger=None):
    assert os.path.exists(src_patch_level_dataset_type_dir)
    assert os.path.exists(src_radiograph_level_dataset_type_dir)

    patch_level_image_dir = os.path.join(src_patch_level_dataset_type_dir, 'images')
    patch_level_filename_list = os.listdir(patch_level_image_dir)
    patch_level_image_num = len(patch_level_filename_list)
    assert patch_level_image_num > 0

    radiograph_level_image_dir = os.path.join(src_radiograph_level_dataset_type_dir, 'images')
    radiograph_level_filename_list = os.listdir(radiograph_level_image_dir)
    radiograph_level_image_num = len(radiograph_level_filename_list)
    assert radiograph_level_image_num > 0

    sub_patch_level_filename_list = list()
    for patch_level_filename in patch_level_filename_list:
        for radiograph_level_filename in radiograph_level_filename_list:
            radiograph_prefix = radiograph_level_filename.split('.')[0]
            if radiograph_prefix in patch_level_filename:
                sub_patch_level_filename_list.append(patch_level_filename)

    sub_patch_level_filename_list = list(set(sub_patch_level_filename_list))
    sub_patch_level_image_num = len(sub_patch_level_filename_list)

    if logger is None:
        print('*******************************************************************************************************')
        print('Source {} dataset contains {} {}.'.format(dataset_type, patch_level_image_num, patch_type))
        print('Destination {} sub-dataset-{} contains {} {}.'.format(dataset_type, sub_dataset_idx + 1,
                                                                     sub_patch_level_image_num, patch_type))
    else:
        logger.write_and_print('**************************************************************************************')
        logger.write_and_print(
            'Source {} dataset contains {} {}.'.format(dataset_type, patch_level_image_num, patch_type))
        logger.write_and_print('Destination {} sub-dataset-{} contains {} {}.'.format(dataset_type, sub_dataset_idx + 1,
                                                                                      sub_patch_level_image_num,
                                                                                      patch_type))

    return sub_patch_level_filename_list


def copy_data_from_src_2_dst(src_patch_level_dataset_type_dir, dst_patch_level_dataset_type_dir,
                             sub_patch_level_filename_list, sub_dataset_idx, dataset_type, patch_type, logger=None):
    assert os.path.exists(src_patch_level_dataset_type_dir)
    assert os.path.exists(dst_patch_level_dataset_type_dir)

    # copy images and labels
    file_idx = 0
    for filename in sub_patch_level_filename_list:
        file_idx += 1

        src_absolute_image_path = os.path.join(src_patch_level_dataset_type_dir, 'images', filename)
        dst_absolute_image_path = os.path.join(dst_patch_level_dataset_type_dir, 'images', filename)
        shutil.copy(src_absolute_image_path, dst_absolute_image_path)

        src_absolute_label_path = os.path.join(src_patch_level_dataset_type_dir, 'labels', filename)
        dst_absolute_label_path = os.path.join(dst_patch_level_dataset_type_dir, 'labels', filename)
        shutil.copy(src_absolute_label_path, dst_absolute_label_path)

        if logger is None:
            print('  Sub {4} dataset-{0} contains {2} {3}, copying data {1} out of {2}.'.format(sub_dataset_idx + 1,
                                                                                                file_idx,
                                                                                                len(sub_patch_level_filename_list),
                                                                                                patch_type,
                                                                                                dataset_type))
        else:
            logger.write_and_print(
                '  Sub {4} dataset-{0} contains {2} {3}, copying data {1} out of {2}.'.format(sub_dataset_idx + 1,
                                                                                              file_idx,
                                                                                              len(sub_patch_level_filename_list),
                                                                                              patch_type,
                                                                                              dataset_type))

    return
