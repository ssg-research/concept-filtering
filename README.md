# Espresso

This paper will appear in ACM Conference on Data and Application Security and Privacy (CODASPY) 2025.

This repo contains code that allows you to reproduce experiments for the concept-removal technique presented in "Espresso: Robust Concept Filtering in Text-to-Image Models". An extended version of the paper is available on [ArXiv](https://arxiv.org/abs/2404.19227).

## Requirements

You need __conda__. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate ldm
```

To update the env:

```bash
conda env update --name ldm --file environment.yml
```

or

```bash
conda activate ldm
conda env update --file environment.yml
```

Once activated, go to the stable-diffusion/ library and install it as a package using the following command

```bash
pip install -e .
```

## Dataset

Please ensure that the base_clip.pt is available in the root of this project. You can find it [here](https://zenodo.org/records/13737646)


## Usage

Prior to working in this repo, please remember to activate your environment.
```bash
conda activate ldm
```

### Train and Save Finetuned Models

First, stable-diffusion/generate_imgs.sh must be run to generate the training, validation, and testing images. The prompts for these images are located in stable-diffusion/assets/finetune_prompts, for the training and validation prompts, and stable-diffusion/assets/eval_prompts for the test prompts. The validation are of the form \*_val.txt. The script stable-diffusion/generate_imgs.sh can be used to generate all of these images. Please ensure that the input text prompt files are set in accordance with the desired concept. For hateful, nudity, disturbing, and violent, the testing prompts are i2p prompts, and their file names are of the form i2p_\*.txt.

The following is an example invocation within stable-diffusion/generate_imgs.sh

```bash
input="assets/i2p_violence_prompts.txt"

while IFS= read -r line;
do
    python scripts/txt2img.py --prompt "$line" --plms --experiment violence --outdir_suffix baseline --ourclip_ckpt /u1/test/generative-content-removal/model_checkpoints_final_april/L-patch-14/preserve/model_2_ce_violence_counter_take1_loss1356_te.pt
done < "$input"
```
Use the --no_filter option to suppress the use of a filter. 
The --outdir_suffix option allows for the specification of a suffix for the directory which will store the final images. 
The --ourclip_ckpt option allows use to use a CLIP text model as the Espresso filter. Prior to training, this option should not be set.
The input specifies the prompt file.



After that, the following script can be used to train the Espresso models, given the aforementioned training images. 
```bash
./train_clip_models.sh
```

The stable-diffusion/generate_imgs.sh file can then be invoked for the testing prompts, setting each input prompt file accordingly.
Once again, the directory stable-diffusion/assets/eval_prompts contains the evaluation prompts. 


### Robustness Testing
In order to run the robustness tests, please run the desired attack in the attacks/ directory. The resulting adversarial prompt files can be used in the scripts provided in pez_based_attacks/ to generate the adversarial images. Each CRT has a folder called pez_based_attacks/ which contains scripts to run generation for each attacks' adversarial prompt files.

Use the test data in order to generate the adversarial prompts/images.

Note: For the typographic attack, stable-diffusion/scripts/typographic_attack.sh needs to be run first, in order to generate the images 
required for this attack.
