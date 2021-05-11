CUDA_VISIBLE_DEVICES=1 python test.py --cGAN --data_root data/ \
                --calc_FID --batch_size 128 \
                --results_root results_transform_class_embeddings \
                --transform_space embeddings \
                --n_eval_batches 200 \
                --args_path results_transform_class_embeddings/cGAN/210508_2048/args.json \
                --gen_ckpt_path results_transform_class_embeddings/cGAN/210508_2048/gen_13_iter_0013000.pth.tar \
                --dis_ckpt_path results_transform_class_embeddings/cGAN/210508_2048/dis_13_iter_0013000.pth.tar \
                --n_fid_images 50000
