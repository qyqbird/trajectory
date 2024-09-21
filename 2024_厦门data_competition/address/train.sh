CUDA_VISIBLE_DEVICES=0 python  ../main.py --task aspect \
                  --model_dir address_model \
                  --model_type macbertcrf \
                  --slot_label_file address_slot.txt \
                  --learning_rate 0.00004 \
                  --use_crf \
                  --num_train_epochs 7 \
                  --do_train \
                  --save_steps 80


# python  ../main.py --task aspect \
#                    --model_dir address_model \
#                    --model_type macbertcrf \
#                    --slot_label_file address_slot.txt \
#                    --learning_rate 0.00004 \
#                    --use_crf \
#                    --num_train_epochs 2 \
#                    --do_train \
#                    --save_steps 108
#本地debug
#/Users/buring/study/JointBERT/venv/bin/python ../main.py --task aspect \
#                  --model_dir aspect_model \
#                  --use_crf \
#                  --num_train_epochs 8 \
#                  --do_train \
