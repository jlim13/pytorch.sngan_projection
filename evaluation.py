import os

import numpy as np
import torchvision

import metrics.fid
import utils


def prune_dict(real_dict):
    pruned_dict = {x: [] for x in range(len(real_dict.keys()))}

    for k,v in real_dict.items():
        flattened_vals =  [item for sublist in v for item in sublist]
        flattened_vals = np.asarray(flattened_vals)

        pruned_dict[k] = flattened_vals

    return pruned_dict


def evaluate(args, current_iter, gen, device,
             inception_model=None, eval_iter=None,to_save=False):
    """Evaluate model using 100 mini-batches."""
    calc_fid = (inception_model is not None) and (eval_iter is not None)
    num_batches = args.n_eval_batches

    gen.eval()
    fake_list, real_list = [], []
    conditional = args.cGAN

    class_fake_dict = {x: [] for x in range(args.num_classes)}
    class_real_dict = {x: [] for x in range(args.num_classes)}

    for i in range(1, num_batches + 1):
        if conditional:
            class_id = i % args.num_classes
        else:
            class_id = None
        fake = utils.generate_images(
            gen, device, args.batch_size, args.gen_dim_z,
            args.gen_distribution, class_id=class_id
        )

        if calc_fid and i <= args.n_fid_batches:

            real_data_sample = next(eval_iter)

            for real_class_label in range(args.num_classes):
                real_labels = real_data_sample[1].cpu().numpy()
                these_real_labels = real_labels[real_labels == real_class_label]
                these_real_ims = real_data_sample[0].cpu().numpy()[real_labels == real_class_label]
                class_real_dict[real_class_label].append(these_real_ims)

            real_list.append((real_data_sample[0].cpu().numpy() + 1.0) / 2.0)
            class_fake_dict[class_id].append((fake.cpu().numpy() + 1.0) / 2.0)
            fake_list.append((fake.cpu().numpy() + 1.0) / 2.0)

        if to_save:
            # Save generated images.
            root = args.eval_image_root

            if conditional:
                root = os.path.join(root, "class_id_{:04d}".format(i))
            if not os.path.isdir(root):
                os.makedirs(root)
            fn = "image_iter_{:07d}_batch_{:04d}.png".format(current_iter, i)
            torchvision.utils.save_image(
                fake, os.path.join(root, fn), nrow=4, normalize=True, scale_each=True
            )

    #prune dicts
    class_real_dict = prune_dict(class_real_dict)
    class_fake_dict = prune_dict(class_fake_dict)

    #calc intra-FID scores

    for class_idx in range(args.num_classes):
        real_images = class_real_dict[class_idx]
        fake_images = class_fake_dict[class_idx]

        print ("Class Number: {} | Number of real images {}. Number of fake images {}".format(class_idx, len(real_images), len(fake_images)))
        mu_fake, sigma_fake = metrics.fid.calculate_activation_statistics(
            fake_images, inception_model, args.batch_size, device=device
        )
        mu_real, sigma_real = metrics.fid.calculate_activation_statistics(
            real_images, inception_model, args.batch_size, device=device
        )
        fid_score = metrics.fid.calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
        print ("Class Label {} || Fid Score {}".format(class_idx, fid_score))
    

    # Calculate FID scores
    if calc_fid:
        fake_images = np.concatenate(fake_list)
        real_images = np.concatenate(real_list)
        print ("Number of real images {}. Number of fake images {}".format(len(real_images), len(fake_images)))
        mu_fake, sigma_fake = metrics.fid.calculate_activation_statistics(
            fake_images, inception_model, args.batch_size, device=device
        )
        mu_real, sigma_real = metrics.fid.calculate_activation_statistics(
            real_images, inception_model, args.batch_size, device=device
        )
        fid_score = metrics.fid.calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
    else:
        fid_score = -1000
    gen.train()
    return fid_score
