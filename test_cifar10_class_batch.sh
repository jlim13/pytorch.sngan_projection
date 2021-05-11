CUDA_VISIBLE_DEVICES=0 python test.py --cGAN --data_root data/ \
                --calc_FID --batch_size 128 \
                --results_root results_transform_batch_labels \
                --n_eval_batches 200 \
                --args_path results_transform_batch_labels/cGAN/210509_2324/args.json \
                --gen_ckpt_path results_transform_batch_labels/cGAN/210509_2324/gen_12_iter_0012000.pth.tar \
                --dis_ckpt_path results_transform_batch_labels/cGAN/210509_2324/dis_12_iter_0012000.pth.tar \
                --n_fid_images 50000



# --args_path results_transform_batch_labels/cGAN/210509_0316/args.json \
