CUDA_VISIBLE_DEVICES=0 python train_64.py --cGAN --data_root data/ \
                --calc_FID --batch_size 128 \
                --results_root results_oversample \
                --oversample 1
