
        


export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="stable-diffusion/jan6-nudity-12-loss13456--noti-_i2p-baseline/samples"
export SAFETY_CONCEPT='nude'
export OUTPUT_DIR="results_sdd_adv_attempt2_{$SAFETY_CONCEPT}_i2p_normalclip_sddckpt/"

accelerate launch concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="object" \
        --placeholder_token="<s1>" --initializer_token="nude" \
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
        --ac_checkpoint="/u1/test/safe-diffusion/saved/nudity/unet/diffusion_pytorch_model.safetensors" \
        --mixed_precision="fp16" \
        --nud_prompt "nude" \
        --clean_prompt "clean" \
        