#CUDA_VISIBLE_DEVICES=1 python ../predict.py --input_file "data/blind" \
#                  --model_dir "aspect_model" \
#                  --use_crf \
#                  --output_file "data/predict"

python ../predict.py --input_file "data/blind" \
                  --model_dir "address_model" \
                  --use_crf \
                  --output_file "data/predict"