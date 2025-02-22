export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export DATA_DIR="stable-diffusion/outputs/jan3-violence-12-loss1356--noti/samples"
export DATA_DIR="stable-diffusion/jan6-violence-None-None--noti-__i2p-baseline/samples"
export SAFETY_CONCEPT='violence'
export OUTPUT_DIR="results_esd_adv_attempt_{$SAFETY_CONCEPT}_i2p_normalclip/"
export ESD_CKPT="stable-diffusion-copy/stable-diffusion/models/diffusers-word_violent-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_violent-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="object" \
        --placeholder_token="<s1>" --initializer_token="r" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=5000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=5000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=25 \
        --esd_checkpoint=$ESD_CKPT \
        --mixed_precision="fp16" \   
        --nud_prompt "violent" \
        --clean_prompt "peaceful" \

