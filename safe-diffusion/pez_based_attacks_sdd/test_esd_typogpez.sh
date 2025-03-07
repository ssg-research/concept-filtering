#!/bin/bash
    # --prompt_path attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_nudity_normalclip_.txt \

python generate.py --model_name='/u1/test/safe-diffusion/saved/nudity' \
    --save_path 'outputs_typogpez_i2p/stanford_sdd_nudity' \
    --num_samples 10 \
    --pipeline_type sdd --use_fp16 
    --prompt_path attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_nudity_i2p_normalclip_nonadaptive__sampled200.txt






# python generate.py --model_name='/u1/test/safe-diffusion/saved/marvel' \
#     --prompt_path attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_marvel_normalclip_.txt\
#     --save_path 'outputs_typogpez/stanford_sdd_marvel' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 


    


# python generate.py --model_name='/u1/test/safe-diffusion/saved/nemo' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_nemo_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_nemo' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/grumpy' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_grumpy_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_grumpy_cat' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/r2d2' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_r2d2_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_r2d2' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/snoopy' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_snoopy_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_snoopy' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/jolie' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_jolie_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_jolie' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/swift' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_swift_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_swift' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/musk' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_musk_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_musk' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/pitt' \
#     --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_pitt_normalclip_.txt' \
#     --save_path 'outputs_typogpez/stanford_sdd_pitt' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 




    # --prompt_path 'attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_violence_normalclip_.txt' \

python generate.py --model_name='/u1/test/safe-diffusion/saved/violence' \
    --save_path 'outputs_typogpez_i2p/stanford_sdd_violence' \
    --num_samples 10 \
    --pipeline_type sdd --use_fp16 
    --prompt_path attacks/hard-prompts-made-easy/adv_prompts_typogattack_on_violence_i2p_normalclip_nonadaptive__sampled200.txt

