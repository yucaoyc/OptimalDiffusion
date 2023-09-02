import os
import torch
import numpy as np
from scipy import linalg
from lib.inception import InceptionV3
from lib.fid_score import *
from lib.generate_batch import *
from lib.helpers import create

def load_v3model(use_fid_inception=True):
    """
        Import the inception v3 model.
        If use_fid_inception=True, use tensorflow version.
        If use_fid_inception=False, use torchvision version.
    """

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    v3model = InceptionV3([block_idx], use_fid_inception=use_fid_inception)
    return v3model

"""
    mu is a vector
    return mu \otimes mu (namely, the outer product of mu)
"""
def vvt(mu):
    dims = len(mu)
    return np.matmul(np.reshape(mu, (dims,1)), np.transpose(np.reshape(mu,(dims,1))))

"""
    Adapted from pytorch-fid.
    We include the second moment instead.

    Remark:
    - sigma is biased, namely, we use 1/N instead of 1/(N-1) to compute average.
    - same for second moment.
"""
def calculate_activation_statistics(batch, model, dims=2048):

    model.eval()
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    # if input is 60*3*32*32, then output of pred is 60*2048*1*1
    # that is why we need to reshape it.
    act = pred.squeeze(3).squeeze(2).cpu().numpy()

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False, bias=True)
    secm = sigma + vvt(mu)

    return mu, sigma, secm



def load_precomputed_stats(pre_filename):
    """
    If we have already computed statistics, we simply load statistics.
    """
    if os.path.exists(pre_filename):
        # Load data
        trainset_mu, trainset_sigma = torch.load(pre_filename)
        return trainset_mu.numpy(), trainset_sigma.numpy()



def generated_stats(gen_sde, model, device, \
        alpha, logit, args, num_sample, batch_size, \
        trans_resize_toimg, img_size, \
        num_method = "ei", \
        num_steps=200, path_of_data="", dims=2048, verbose=False, save_img=False, \
        save_batch_statistics=True):
    """
        Given a model gen_sde, and a v3model (called model herein), we compute statistics
        needed for FID.

        By default, we don't save images, as it takes a large amount of time in i/o.
        We simply compute statistics and only save mean and covariance directly after
        we generate images.

        It has some side effect:

        when image size is small and batch_size is small, this function might utilize
        more hard disk space than simply saving images, as the v3 model's output size is
        2048 by default. You may pass save_batch_statistics = False to save hard disk
        space, at the cost of not being able to resume the checkpoint.

        todo: we only need dataset size information inside args,
        consider removing the dependence of args later.
    """

    final_filename = os.path.join(path_of_data,
        "gen_img_{:s}_{:d}_{:3d}_{:d}___{:d}.npy".format(num_method, num_steps, \
            int(alpha*100), batch_size, num_sample))

    if os.path.exists(final_filename):
        # we have already compute all data, just return it
        with open(final_filename, 'rb') as f:
            mu = np.load(f)
            sigma = np.load(f)
            return mu, sigma

    # otherwise, let do the calculation
    final_mu = np.zeros(dims)
    final_secm = np.zeros((dims,dims))
    existing_size = 0
    batch_idx = 0

    while existing_size < num_sample:
        count_x = batch_size
        # check if the cache file already exists
        filename = os.path.join(path_of_data, "gen_img_{:s}_{:d}_{:3d}_{:d}_{:d}.npy".format(num_method, \
                                            num_steps, int(alpha*100), batch_size, batch_idx))

        if os.path.exists(filename):
            # we load the statistics
            # Remark:
            #   cache file saves mu and secm
            #   final data files saves mu and cov.

            with open(filename, 'rb') as f:
                mu = np.load(f)
                secm = np.load(f)
        else:
            # we should generate the data
            y0 = generate_batch_sample(gen_sde, alpha, logit, args, \
                num_method=num_method, num_steps=num_steps, num_samples = batch_size, verbose=verbose)

            # save image and then resize images
            if batch_idx == 0 and save_img:
                path_of_img = os.path.join(path_of_data, "gen_img_{:s}_{:d}_{:3d}_{:d}_{:d}/".format(num_method, \
                                            num_steps, int(alpha*100), batch_size, batch_idx))
                create(path_of_img)
                # save the first batch of images as samples for testing
                y0_new = save_batch_sample(y0, path_of_img, trans_resize_toimg, \
                                           new_size = img_size, start_idx = existing_size, \
                                           verbose=verbose, save_img=True, max_img_size=20)
            else:
                y0_new = save_batch_sample(y0, "", trans_resize_toimg, \
                                           new_size = img_size, start_idx = existing_size, \
                                           verbose=verbose, save_img=False)

            # compute statistics used in FID
            mu, _, secm = calculate_activation_statistics(y0_new, model)
            # save statistics
            if save_batch_statistics:
                with open(filename, 'wb') as f:
                    np.save(f, mu)
                    np.save(f, secm)

        # Update the total mu and sigma
        final_mu = (final_mu * existing_size + count_x * mu)/(existing_size + count_x)
        final_secm = (final_secm * existing_size + count_x * secm)/(existing_size + count_x)
        existing_size = existing_size + count_x
        batch_idx = batch_idx + 1

        if verbose:
            print("Existing size = {:d}".format(existing_size))

    # Compute covariance matrix
    final_sigma = final_secm - vvt(final_mu)
    # correct bias
    final_sigma = final_sigma * existing_size/(existing_size-1)

    # save final statistics for later use
    with open(final_filename, 'wb') as f:
        np.save(f, final_mu)
        np.save(f, final_sigma)

    return final_mu, final_sigma
