import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

data = pd.read_pickle('fid.pkl')

def find_fid_value(data, train_iter, seed, weight, mixed):
    """
        data means pd.read_pickle('fid.pkl')
    """
    value = []
    for ind in data.index:
        ckpt_path = data.checkpoint[ind]

        # specify file name
        if train_iter != 200000:
            ckpt_fn = 'checkpoint-iter-{:d}.pt'.format(train_iter)
        else:
            ckpt_fn = 'checkpoint.pt'

        # check if filename matches
        if ckpt_fn in ckpt_path and \
            '{:d}-True-{:s}'.format(seed, str(mixed)) in ckpt_path and \
            'cifar-{:s}'.format(weight) in ckpt_path:
            value.append(data.FID[ind])

    if len(value) == 0:
        warnings.warn("data not found")
    elif len(value) > 1:
        warnings.warn("duplicate items")

    return value

def find_sml_value(train_iter, seed, weight, mixed, nt):
    if train_iter != 200000:
        ckpt_fn = 'checkpoint-iter-{:d}.pt'.format(train_iter)
    else:
        ckpt_fn = 'checkpoint.pt'

    ckpt_path = "saved/cifar-{:s}-64-200000-1.0-{:d}-True-{:s}/".format(weight,seed,str(mixed))+ckpt_fn+'_{:d}_sml.pkl'.format(nt)

    data_dict = pickle.load(open(ckpt_path,'rb'))
    return data_dict


train_iter_list = np.array([40000, 80000, 120000, 160000, 200000])
train_iter_label = ['40k', '80k', '120k', '160k', '200k']
seed_list = [0, 1]
weight_list = ['data','noise','default']
mixed_list = [True, False]
T0 = 1.0

#################################################
# Plot FID
fig = plt.figure(figsize=(8,5))
for seed_ind in range(2):
    for mixed_ind in range(2):

        seed = seed_list[seed_ind]
        mixed = mixed_list[mixed_ind]
        fig_idx = seed_ind*2 + mixed_ind + 1

        plt.subplot(2, 2, fig_idx)
        for weight in weight_list:
            fid_array = []
            for train_iter in train_iter_list:
                fid_array.append(find_fid_value(data, train_iter, seed, weight, mixed))

            plt.plot(train_iter_list/1000, fid_array, 'x-', label="{:s}".format(weight))
            plt.xticks(ticks=train_iter_list/1000, labels=train_iter_label)
            plt.title("trial={:d}, mixed={:s}".format(seed, str(mixed)), fontsize=12)
            plt.yscale('log')
        plt.ylabel("FID", fontsize=12)
        if fig_idx == 4:
            plt.legend(bbox_to_anchor=(1.0,1.0), fontsize=12)
        plt.tight_layout()

        print([seed, mixed, weight, fid_array[-1]])

plt.savefig('../../assets/cifar-fid.eps')

##################################################
# Plot SML
nt = 50

fig = plt.figure(figsize=(8,5))
for seed_ind in range(len(seed_list)):
    for mixed_ind in range(len(mixed_list)):
        seed = seed_list[seed_ind]
        mixed = mixed_list[mixed_ind]
        fig_idx = seed_ind*2 + mixed_ind + 1
        plt.subplot(2, 2, fig_idx)
        for weight in weight_list:
            sml_array = []
            for train_iter in train_iter_list:
                sml_array.append(find_sml_value(train_iter, seed, weight, mixed, nt)['sml'])
            plt.plot(train_iter_list/1000, sml_array, 'x-', label="{:s}".format(weight))
            plt.xticks(ticks=train_iter_list/1000, labels=train_iter_label)
            plt.title("trial={:d}, mixed={:s}".format(seed, str(mixed)), fontsize=12)
        plt.ylabel('SML',fontsize=12)
        if fig_idx == 4:
            plt.legend(bbox_to_anchor=(1.6,1.0), fontsize=12)
        plt.tight_layout()
plt.savefig("../../assets/cifar_sml.eps")

##################################################
# Plot error distribution
for seed in seed_list:
    for mixed in mixed_list:
        f1 = plt.figure(figsize=(7,9))
        for train_iter_ind in range(len(train_iter_list)):
            train_iter = train_iter_list[train_iter_ind]
            t_vec = np.linspace(T0/nt, T0, nt, endpoint=True)
            l_vec_noise = find_sml_value(train_iter, seed, 'noise', mixed, nt)['smlvec']
            l_vec_default = find_sml_value(train_iter, seed, 'default', mixed, nt)['smlvec']
            l_vec_data = find_sml_value(train_iter, seed, 'data', mixed, nt)['smlvec']
            plt.subplot(5,2,train_iter_ind*2+1)
            plt.plot(t_vec/T0, l_vec_noise/l_vec_default, 'k-', label="noise")
            plt.plot(t_vec/T0, l_vec_data/l_vec_default, 'b-', label="data")
            if train_iter_ind == 0:
                plt.title("Relative loss", fontsize=12)
            plt.ylabel("train step = "+train_iter_label[train_iter_ind], fontsize=12)
            plt.ylim([0.5, 2])
            if train_iter_ind == 4:
                plt.xlabel("t/T", fontsize=12)
            plt.subplot(5,2,train_iter_ind*2+2)
            plt.plot(t_vec/T0, l_vec_noise, 'k-', label="noise")
            plt.plot(t_vec/T0, l_vec_data, 'b-', label="data")
            plt.plot(t_vec/T0, l_vec_default, 'r-', label="default")
            plt.yscale("log")
            if train_iter_ind == 0:
                plt.title("Loss", fontsize=12)
                plt.legend(fontsize=12, bbox_to_anchor=(1.0,1.0))
            if train_iter_ind == 4:
                plt.xlabel("t/T", fontsize=12)
        plt.suptitle("trial={:d}, mixed={:s}".format(seed, str(mixed)), fontsize=16)
        plt.tight_layout()
        plt.savefig("../../assets/cifar-{:d}-{:s}.eps".format(seed, str(mixed)))
