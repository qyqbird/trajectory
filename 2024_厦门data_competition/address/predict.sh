CUDA_VISIBLE_DEVICES=0 python ../predict.py --input_file "data/blind" \
                 --model_dir "address_model" \
                 --model_type macbertcrf \
                 --use_crf \
                 --output_file "data/prelim_submit"

# python ../predict.py --input_file "data/blind" \
#                   --model_dir "address_model" \
#                   --use_crf \
#                   --output_file "data/prelim_submit"