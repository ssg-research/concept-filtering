import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
import torchvision.transforms as transforms
from contextlib import contextmanager, nullcontext
import copy

from ldm.util import instantiate_from_config, generate_typo
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor, CLIPModel, CLIPProcessor

# from optim_utils import optimize_prompt
from omegaconf import OmegaConf
# from src_fmn.train_ti import train as ti_component
# from src_fmn.simple_inference import main as test_sampling


# from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
import clip

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    # try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        print('Loading rick')
        return y
    # except Exception:
        # return x


def check_safety(x_image):
    # safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    # x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    # assert x_checked_image.shape[0] == len(has_nsfw_concept)
    # for i in range(len(has_nsfw_concept)):
    #     if has_nsfw_concept[i]:
    #         x_checked_image[i] = load_replacement(x_checked_image[i])
    # return x_checked_image, False
    return x_image, False


def evaluate_with_filter(opt, model_finetuned, processor, image_list, base_count, device):
    noreg = ''
    images_to_ret = []
    if opt.noreg:
        noreg = 'noreg_'
    

    # prompts_with_nud = f'{learned_prompt} naked'
    if opt.experiment == 'nudity':
        prompts_with_nud = f'nude'
        prompts_with_clean = f"clean"

    if opt.experiment == 'grumpy':
        prompts_with_nud = 'grumpy cat'
        prompts_with_clean = 'cat'

    elif opt.experiment == 'r2d2':
        prompts_with_nud = 'r2d2'
        prompts_with_clean = 'robot'

    elif opt.experiment == 'marvel':
        prompts_with_nud = 'captain marvel'
        prompts_with_clean = 'female superhero'

    elif opt.experiment == 'nemo':
        prompts_with_nud = 'nemo'
        prompts_with_clean = 'fish'

    elif opt.experiment == 'musk':
        prompts_with_nud = 'Elon Musk'
        prompts_with_clean = 'man'

    elif opt.experiment == 'pitt':
        prompts_with_nud = 'Brad Pitt'
        prompts_with_clean = 'man'

    elif opt.experiment == 'swift':
        prompts_with_nud = 'Taylor Swift'
        prompts_with_clean = 'woman'
    
    elif opt.experiment == 'jolie':
        prompts_with_nud = 'Angelina Jolie'
        prompts_with_clean = 'woman'
    
    elif opt.experiment == 'snoopy':
        prompts_with_nud = 'Snoopy'
        prompts_with_clean = 'dog'
    
    elif opt.experiment == 'violence' or opt.experiment == 'peaceful':
        prompts_with_nud = 'violent'
        prompts_with_clean = 'peaceful'

    for img_num, img in enumerate(image_list):

        inputs = processor(text=[prompts_with_nud, prompts_with_clean], images=[img], return_tensors="pt", padding=True).to(device)
        outputs = model_finetuned(**inputs)
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        print(f"CLIP accuracy [{prompts_with_nud}, {prompts_with_clean}]:", probs)

        if probs[0][0] > probs[0][1]:
        # if sim_to_nud > sim_to_clean:
            print("Nud detected")
            if not opt.no_filter:
                img = Image.open('assets/rick.jpeg')
        images_to_ret.append(img)

        base_count += 1


    return images_to_ret

def run_checks(opt, model_finetuned, processor, sample_path_ti, x_checked_image_torch, base_count, device, do_eval = True):
    noreg = ''
    if opt.noreg:
        noreg = 'noreg_'

    image_list = []
    for img_num, x_sample in enumerate(x_checked_image_torch):
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        image_list.append(img)
        
        if opt.experiment is not None:
            print(f'path: {os.path.join(sample_path_ti, f"{opt.start_at}_{img_num:05}_{opt.experiment}_{noreg}.png")}')
            img.save(os.path.join(sample_path_ti, f"{opt.start_at}_{img_num:05}_{opt.experiment}_{noreg}.png"))
        else:
            print(f'path: {os.path.join(sample_path_ti, f"{opt.start_at}_{img_num:05}.png")}')
            img.save(os.path.join(sample_path_ti, f"{opt.start_at}_{img_num:05}.png"))
        


    if do_eval:
        return evaluate_with_filter(opt, model_finetuned, processor, image_list, base_count, device)

    return image_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        # default="configs/stable-diffusion/nudity_takeall_fulloss_noreg.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        '--ckpt_disregard',
        help='Dummy'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='The concept that we are testing for',
        default = None
    )
    parser.add_argument(
        '--noreg',
        action='store_true',
        default=False,
        help='Text encoder should be reg or noreg. If noreg, use this argument'
    )
    # parser.add_argument(
    #     '--contr',
    #     action='store_true',
    #     default=False,
    #     help='Use the text encoders partially trained with contrastive loss'
    # )
    parser.add_argument(
        '--testing_coco',
        action = 'store_true',
        default = False,
        help='Use this argument if testing on the coco dataset. The name of the outdir will be set accordingly'
    )
    parser.add_argument('--typo_check',
                        action = 'store_true',
                        default = False,
                        help = 'Toggle this option if the target concept should have typos')
    parser.add_argument('--model_dir',
                        default = '',
                        type=str,
                        help='The model directory in L-patch-14/ where the models are stored'
                        )
    parser.add_argument('--adv_trained',
                        action='store_true',
                        default=False,
                        help='Toggle this option if the model being evaluated is an adversarially trained model'
                        )
    parser.add_argument('--ti_tensors_path', default = None, help='If using TI, specify the path with this argument')
    parser.add_argument('--ourclip', action='store_true', default=False)
    parser.add_argument('--ourclip_ckpt', type=str, default=None)
    parser.add_argument('--ti_yaml', type=str, default='ti_nud.yaml')
    parser.add_argument('--outdir_suffix', default='', type=str, help = 'Add suffix from the command line for the output directory for the results')
    parser.add_argument('--no_filter', action='store_true', default=False)
    parser.add_argument('--start_at', type=int, default=0)
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")

    # If we are running the experiment, there will be other values defined in the arguments which specify params for 
    #    the clip text encoder
    adv_suffix = ''
    if opt.adv_trained:
        adv_suffix = '_adv'
    print(opt.experiment)
    if opt.experiment is not None:
        opt.outdir = f'nov-{opt.experiment}-seed{opt.seed}-{opt.outdir_suffix}'
        if opt.testing_coco:
            opt.outdir += '-coco'

    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    print(f'outpath: {outpath}')

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_path_ti = os.path.join(outpath, "ti_data")
    os.makedirs(sample_path_ti, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    wrapped_model, preprocess = clip.load("ViT-L/14")
    wrapped_model = wrapped_model.to(device)

    args_oc = argparse.Namespace()
    args_oc_dict = {
        "prompt_len": 16,
        "iter": 1000,
        "lr": 0.1,
        "weight_decay": 0.1,
        "prompt_bs": 1,
        "loss_weight": 1.0,
        "print_step": 100,
        "batch_size": 1,
        "clip_model": "ViT-L-14",
        "clip_pretrain": "openai"
    }
    args_oc.__dict__ = args_oc_dict
    
    noreg = ''
    if opt.noreg:
        noreg = 'noreg_'
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""], save_emb=1)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts, save_emb=1)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                            
                            
                        model_finetuned = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
                        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

                        # ourclip = False
                        # ourclip = opt.ourclip
                        if opt.ourclip_ckpt is not None and not opt.no_filter:
                            print(f'ourclip_ckpt: {opt.ourclip_ckpt}')
                            te_checkpoint = torch.load(opt.ourclip_ckpt)
                            model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
                            # model_finetuned.vision_model.load_state_dict(te_checkpoint['visionEncoder_state_dict'])

                        final_images = run_checks(opt, model_finetuned, processor, sample_path_ti, x_checked_image_torch, base_count, device, do_eval=(not opt.no_filter))

                        if not opt.skip_save:
                            for img_num, img in enumerate(final_images):
                                img = put_watermark(img, wm_encoder)
                                if opt.experiment is not None:
                                    print(f'path: {os.path.join(sample_path, f"{base_count:05}_{noreg}.png")}')
                                    img.save(os.path.join(sample_path, f"{base_count:05}_{noreg}.png"))
                                else:
                                    print(f'path: {os.path.join(sample_path, f"{base_count:05}.png")}')
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1


                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
