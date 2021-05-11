CUDA_VISIBLE_DEVICES=3 python train_64.py --cGAN --data_root data/ \
                --calc_FID --batch_size 128 \
                --results_root results_transform_class_embeddings \
                --transform_space embeddings
