import os
import sys
from omegaconf import OmegaConf
from src.train_ti import train as ti_component
from src.train_attn import main as attn_component
from src.simple_inference import main as test_sampling

conf_path = sys.argv[1]
conf = OmegaConf.load(conf_path)

patch_ti = None
multi_concept = None
output_dir = None
prompts_path = None
seed = 1
if "Ti" in conf:
    patch_ti = conf.Ti
    output_dir = conf.Ti.output_dir
    ti_component(**conf.Ti)
    OmegaConf.save(config=conf, f=f"{output_dir}/configs.yaml")
elif "Attn" in conf:
    multi_concept = conf.Attn.multi_concept
    output_dir = conf.Attn.output_dir.split(',')[0]
    print(output_dir)
    prompts_path = conf.Attn.prompts_path.split(',')[-1]
    seeds = str(conf.Attn.seed).split(',')[-1]
    attn_component(conf.Attn)
    OmegaConf.save(config=conf, f=f"{output_dir}/configs.yaml")
else:
    raise ValueError(f"config file not {conf_path} recognized!")

os.makedirs(output_dir, exist_ok=True)
# for pp, seed in zip(prompts_path, seeds):
test_sampling(OmegaConf.create({
    "pretrained_model_name_or_path": output_dir, # SPLIT WAS ADDED
    "save_dir": output_dir,
    "patch_ti": patch_ti,
    "multi_concept": multi_concept,
    "prompts_path":prompts_path,
    "seed": seed
}))

