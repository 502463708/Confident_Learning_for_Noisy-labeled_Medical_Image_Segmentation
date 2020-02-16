import os
import random
import shutil


def filename_list_split(src_dataset_type_dir, sub_dataset_number, random_seed, logger=None):
    assert os.path.exists(src_dataset_type_dir)
    assert isinstance(sub_dataset_number, int) and sub_dataset_number > 1

    dataset_type = src_dataset_type_dir.split('/')[-1]

    if random_seed >= 0:
        random.seed(random_seed)

    image_dir = os.path.join(src_dataset_type_dir, 'images')
    filename_list = os.listdir(image_dir)
    image_num = len(filename_list)
    assert image_num > 0

    if logger is None:
        print('*******************************************************************************************************')
        print('Source {} dataset contains {} images.'.format(dataset_type, image_num))
    else:
        logger.write_and_print('**************************************************************************************')
        logger.write_and_print('Source {} dataset contains {} images.'.format(dataset_type, image_num))

    random.shuffle(filename_list)

    sub_filename_list_list = list()
    image_num_for_each_sub_dataset = image_num // sub_dataset_number
    for i in range(sub_dataset_number):
        idx_start = i * image_num_for_each_sub_dataset
        idx_end = (i + 1) * image_num_for_each_sub_dataset if i < sub_dataset_number - 1 else image_num

        sub_filename_list = filename_list[idx_start: idx_end]
        sub_filename_list_list.append(sub_filename_list)

    return sub_filename_list_list


def copy_data_from_src_2_dst(src_dataset_type_dir, dst_dataset_type_dir, sub_filename_list, sub_dataset_idx,
                             dataset_type, logger=None):
    assert os.path.exists(src_dataset_type_dir)
    assert os.path.exists(dst_dataset_type_dir)

    # copy images and labels
    data_idx = 0
    for filename in sub_filename_list:
        data_idx += 1

        src_absolute_image_path = os.path.join(src_dataset_type_dir, 'images', filename)
        dst_absolute_image_path = os.path.join(dst_dataset_type_dir, 'images', filename)
        shutil.copy(src_absolute_image_path, dst_absolute_image_path)

        src_absolute_label_path = os.path.join(src_dataset_type_dir, 'labels', filename)
        dst_absolute_label_path = os.path.join(dst_dataset_type_dir, 'labels', filename)
        shutil.copy(src_absolute_label_path, dst_absolute_label_path)

        if logger is None:
            print('  Sub {3} dataset-{0} contains {2} data, copying data {1} out of {2}.'.format(sub_dataset_idx + 1,
                                                                                                 data_idx,
                                                                                                 len(sub_filename_list),
                                                                                                 dataset_type))
        else:
            logger.write_and_print(
                '  Sub {3} dataset-{0} contains {2} data, copying data {1} out of {2}.'.format(sub_dataset_idx + 1,
                                                                                               data_idx,
                                                                                               len(sub_filename_list),
                                                                                               dataset_type))

    return
