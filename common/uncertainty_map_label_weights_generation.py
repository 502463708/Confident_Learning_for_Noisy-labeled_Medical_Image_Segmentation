import os
import SimpleITK as sitk


def save_uncertainty_maps(uncertainty_maps_np, filenames, positive_patch_results_saving_dir,
                          negative_patch_results_saving_dir, logger):
    batch_size = uncertainty_maps_np.shape[0]

    # iterating each image of this batch
    for idx in range(batch_size):
        uncertainty_map_np = uncertainty_maps_np[idx, :, :]
        filename = filenames[idx]
        is_positive_patch = True if 'positive' in filename else False

        logger.write_and_print(
            'Info for the uncertainty map of image {}: max = {:.4f}, min = {:.4f}'.format(filename,
                                                                                          uncertainty_map_np.max(),
                                                                                          uncertainty_map_np.min()))

        uncertainty_map_image = sitk.GetImageFromArray(uncertainty_map_np)

        if is_positive_patch:
            sitk.WriteImage(uncertainty_map_image,
                            os.path.join(positive_patch_results_saving_dir, filename.replace('png', 'nii')))
        else:
            sitk.WriteImage(uncertainty_map_image,
                            os.path.join(negative_patch_results_saving_dir, filename.replace('png', 'nii')))

    return
