import os
import random
import shutil


def random_sample_negative_patches(positive_dir, negative_dir, random_seed, logger=None):
    if random_seed >= 0:
        random.seed(random_seed)

    positive_image_filename_list = os.listdir(positive_dir)
    negative_image_filename_list = os.listdir(negative_dir)

    positive_image_count = len(positive_image_filename_list)
    negative_image_count = len(negative_image_filename_list)

    if logger is None:
        print('Sampling {} negative patches from all of the {} negative patches...'.format(positive_image_count,
                                                                                           negative_image_count))
    else:
        logger.write_and_print(
            'Sampling {} negative patches from all of the {} negative patches...'.format(positive_image_count,
                                                                                         negative_image_count))

    random.shuffle(negative_image_filename_list)
    sampled_negative_image_filename_list = negative_image_filename_list[0:positive_image_count]

    if logger is None:
        print('Finished sampling negative patches...')
    else:
        logger.write_and_print('Finished sampling negative patches...')

    assert len(positive_image_filename_list) == len(sampled_negative_image_filename_list)

    sampled_negative_image_count = positive_image_count

    return positive_image_filename_list, sampled_negative_image_filename_list, positive_image_count, \
           negative_image_count, sampled_negative_image_count


def remove_negative_patches(src_data_root_dir, dst_data_root_dir, dataset_type, random_seed, logger=None):
    assert dataset_type in ['training', 'validation', 'test']

    # generate positive image dir
    positive_image_dir = os.path.join(src_data_root_dir, 'positive_patches', dataset_type, 'images')

    # generate negative image dir
    negative_image_dir = positive_image_dir.replace('positive', 'negative')

    # generate positive_image_filename_list and sampled_negative_image_filename_list
    positive_image_filename_list, \
    sampled_negative_image_filename_list, \
    positive_image_count, \
    negative_image_count, \
    sampled_negative_image_count = random_sample_negative_patches(positive_image_dir, negative_image_dir, random_seed,
                                                                  logger)

    # copy positive images and labels
    idx = 0
    for positive_image_filename in positive_image_filename_list:
        idx += 1
        print('Copying positive patch {} out of {} positive patches...'.format(idx, positive_image_count))

        src_absolute_image_path = os.path.join(positive_image_dir, positive_image_filename)
        dst_absolute_image_path = src_absolute_image_path.replace(src_data_root_dir, dst_data_root_dir)
        shutil.copy(src_absolute_image_path, dst_absolute_image_path)

        src_absolute_label_path = src_absolute_image_path.replace('images', 'labels')
        dst_absolute_label_path = dst_absolute_image_path.replace('images', 'labels')
        shutil.copy(src_absolute_label_path, dst_absolute_label_path)

    # copy negative images and labels
    idx = 0
    for sampled_negative_image_filename in sampled_negative_image_filename_list:
        idx += 1
        print('Copying negative patch {} out of {} negative patches...'.format(idx, sampled_negative_image_count))

        src_absolute_image_path = os.path.join(negative_image_dir, sampled_negative_image_filename)
        dst_absolute_image_path = src_absolute_image_path.replace(src_data_root_dir, dst_data_root_dir)
        shutil.copy(src_absolute_image_path, dst_absolute_image_path)

        src_absolute_label_path = src_absolute_image_path.replace('images', 'labels')
        dst_absolute_label_path = dst_absolute_image_path.replace('images', 'labels')
        shutil.copy(src_absolute_label_path, dst_absolute_label_path)

    return positive_image_count, negative_image_count, sampled_negative_image_count
