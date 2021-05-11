CUDA_VISIBLE_DEVICES=0 python test.py --cGAN --data_root data/ \
                --calc_FID --batch_size 128 \
                --results_root results_normal_cGAN \
                --n_eval_batches 200 \
                --args_path results_normal_cGAN/cGAN/210510_1513/args.json \
                --gen_ckpt_path results_normal_cGAN/cGAN/210510_1513/gen_12_iter_0012000.pth.tar \
                --dis_ckpt_path results_normal_cGAN/cGAN/210510_1513/dis_12_iter_0012000.pth.tar \
                --n_fid_images 50000
