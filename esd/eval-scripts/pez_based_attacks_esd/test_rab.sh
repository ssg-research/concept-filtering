#!/bin/bash

# attacks/Ring-A-Bell/data/InvPrompt/Nudity_3_length_16.csv 
python eval-scripts/generate-images.py --model_name='models/diffusers-nudity-ESDu1-UNET.pt' \
    --prompts_path attacks/Ring-A-Bell/data/InvPrompt_preserve/Nudity_i2p_42_3_length_16_False.csv \
    --save_path 'outputs_rab/stanford_esd_rab_nud_i2p' \
    --num_samples 10 \
    





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_CaptainMarvel-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_CaptainMarvel-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path attacks/Ring-A-Bell/data/InvPrompt/marvel_42_3_length_16_False.csv\
#     --save_path 'outputs_rab/stanford_esd_rab_marvel' \
#     --num_samples 10 \
#     


    


# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_nemo-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_nemo-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/nemo_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_nemo' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_grumpycat-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_grumpycat-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/grumpy_44_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_grumpy_cat' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_r2d2-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_r2d2-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/r2d2_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_r2d2' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_Snoopy-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_Snoopy-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/snoopy_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_snoopy' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_AngelinaJolie-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_AngelinaJolie-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/jolie_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_jolie' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_TaylorSwift-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_TaylorSwift-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/swift_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_swift' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_ElonMusk-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_ElonMusk-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/musk_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_musk' \
#     --num_samples 10 \
#     





# python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_BradPitt-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_BradPitt-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
#     --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt/pitt_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_esd_rab_pitt' \
#     --num_samples 10 \
#     



# attacks/Ring-A-Bell/data/InvPrompt/Violence_3_length_16.csv

python eval-scripts/generate-images.py --model_name='stable-diffusion-copy/stable-diffusion/models/diffusers-word_violent-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_violent-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt' \
    --prompts_path 'attacks/Ring-A-Bell/data/InvPrompt_preserve/Violence_i2p_42_3_length_16_False.csv' \
    --save_path 'outputs_rab/stanford_esd_rab_violence_i2p' \
    --num_samples 10 \
    


