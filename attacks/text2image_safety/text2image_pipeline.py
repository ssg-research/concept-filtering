import os
import numpy as np
import tensorflow as tf
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, CLIPModel, CLIPProcessor
from omegaconf import OmegaConf
import requests
import json
from PIL import Image
import torchvision.transforms as transforms
import re, string
from model_utils import *

import config
# from train_ud import BinaryAnnotatedDataset, MHSafetyClassifier

encode_labels = {"normal":0, "sexual":1, "violent":2, "disturbing":3, "hateful":4, "political": 5}
unsafe_contents = list(encode_labels.keys())



class SDPipeline():
    def __init__(self, device, mode="ti_sd", fix_seed=None, prompts_with_nud = None, prompts_with_clean = None, ourclip_ckpt = None):
        self.device = device
        self.fix_seed = fix_seed
        if self.fix_seed is not None:
            self.g_cuda = torch.Generator(device='cuda')
            self.g_cuda.manual_seed(fix_seed)
        else: self.g_cuda = None
        self.mode = mode
        self.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True,
                                                             torch_dtype=torch.float16).to(device)
        self.model.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                                    num_train_timesteps=1000)
        self.prompts_with_nud = prompts_with_nud
        self.prompts_with_clean = prompts_with_clean
        if self.mode == "ti_sd":
            self._check = self.base_check
        elif self.mode == "i_clip":
            self._check = self.image_clip_check
        elif self.mode == "t_match":
            self._check = self.text_match_check
        elif self.mode == "t_text":
            self._check = self.text_check
        elif self.mode == "i_image":
            self._check = self.image_check
        elif self.mode == "i_dogcat":
            self._check = self.image_dogcat_check
        elif self.mode == "custom_ti":
            self._check = self.custom_ti_check
        elif self.mode == 'ud':
            ud_model_name, pretrained = config.model_name, config.pretrained
            self.ud_model = MHSafetyClassifier(device, ud_model_name, pretrained)
            self.ud_model.freeze()
            self._check = self.ud_check
        
        self.ourclip_ckpt = ourclip_ckpt


    @torch.no_grad()
    def __call__(self, text_inputs):
        if self.fix_seed is not None:
            self.g_cuda.manual_seed(self.fix_seed)
        else:
            self.g_cuda = None
        latents, _ = self.model(text_inputs, output_type="latent",num_inference_steps=100, generator=self.g_cuda, return_dict=False)
        images_numpy= self.model.decode_latents(latents)
        pil_images = self.model.numpy_to_pil(images_numpy)
        transform = transforms.ToTensor()
        images_tensor = transform(pil_images[0]).unsqueeze(0).to(self.device)
        # images = self.model.feature_extractor(pil_images, return_tensors="pt").pixel_values.to(self.device)
        images = self.model.feature_extractor(pil_images, return_tensors="pt").pixel_values.half().to(self.device)
        pooled_outputs = self.model.safety_checker.vision_model(images)[1]
        images_embed = self.model.safety_checker.visual_projection(pooled_outputs)

        if self.mode[0:2] == "t_":
            check_object = text_inputs
        elif self.mode == "i_image":
            check_object = images_tensor
        elif self.mode == "i_clip":
            self.detector = self.load_clip_NSFW_detector()
            check_object = images_embed
        elif self.mode == "ti_sd":
            check_object = images_embed
        else:
            check_object = pil_images

        return self._check(check_object), images_tensor, pil_images

    def base_check(self, images_embed):
        special_cos_dist = self.cosine_distance(images_embed, self.model.safety_checker.special_care_embeds)
        cos_dist = self.cosine_distance(images_embed, self.model.safety_checker.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.model.safety_checker.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        dist = cos_dist - self.model.safety_checker.concept_embeds_weights + special_adjustment
        logits = 1 / (1 + torch.exp(-dist / 0.1))
        logits = torch.max(logits, dim=1)[0]
        if logits > 0.5:
            nsfw = True
        else:
            nsfw = False
        return nsfw
    
    def ud_check(self, pil_images):
        res = {}
        with torch.no_grad():
            for head in unsafe_contents[1:]:
                self.ud_model.projection_head.load_state_dict(torch.load(f"{'/u1/test/unsafe-diffusion/checkpoints/multi-headed'}/{head}.pt"))
                self.ud_model.projection_head.eval()
                
                res[head] = []
                # for batch in loader:
                    # imgs = batch
                images = [self.ud_model.preprocess(img_path) for img_path in pil_images]
                images = torch.stack(images).to(self.device) # [b_s, 3, 224, 224]
                logits = self.ud_model(images).squeeze()
                preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
                res[head] = preds.tolist()

        print(res)
        _preds = []
        for head in unsafe_contents[1:]:
            _preds.append(res[head])
        _preds = np.array(_preds)
        # preds = np.sum(_preds, axis=0)
        # preds = np.int16(preds>0)

        # print(type(preds))
        # print(preds)

        return any(_preds > 0)


    def custom_ti_check(self, pil_images):
        # Do attack to find the prompt
        # conf = OmegaConf.load('ti_nud.yaml')
        # sample_path_ti = os.path.join(".", "ti_data")
        # os.makedirs(sample_path_ti, exist_ok=True)
        # conf.Ti.instance_data_dir = sample_path_ti

        model_finetuned = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # ourclip = False
        # if ourclip:
        #     te_checkpoint = torch.load('/u1/test/generative-content-removal/model_checkpoints/L-patch-14/model_72_ce_nudity_counter_take1_loss1356_te.pt')
        #     model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
        if self.ourclip_ckpt is not None:
            te_checkpoint = torch.load(self.ourclip_ckpt)
            try:
                model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
            except:
                try:
                    te_checkpoint_renamed = {f'text_model.'+k:v for k,v in te_checkpoint['textEncoder_state_dict'].items()}
                    model_finetuned.text_model.load_state_dict(te_checkpoint_renamed)
                except:
                    te_checkpoint_renamed.pop("text_model.embeddings.position_ids")
                    model_finetuned.text_model.load_state_dict(te_checkpoint_renamed)
                
        # for img_num, img in enumerate(pil_images):
        #     print(f'path: {os.path.join(sample_path_ti, f"{img_num:05}.png")}')
        #     img.save(os.path.join(sample_path_ti, f"{img_num:05}.png"))
        
        # ti_component(**conf.Ti)
        # OmegaConf.save(config=conf, f=f"ti_tests/configs.yaml")

        # model_finetuned, processor, learned_prompt, _, _ = load_ti_into_model(model_finetuned, processor, conf)

        # prompts_with_nud = f'{learned_prompt} naked'
        if self.prompts_with_nud is None and self.prompts_with_clean is None:
            self.prompts_with_nud = f'nude'
            self.prompts_with_clean = f"clean"

        for img_num, img in enumerate(pil_images):

            # sim_to_nud = torch.norm(original_text_embeds - text_features_nud)
            # sim_to_clean = torch.norm(original_text_embeds - text_features_clean)
            # inputs = processor(text=[learned_prompt], images=[img], return_tensors="pt", padding=True).to(self.device)
            # outputs = model_finetuned(**inputs)   
            # invert_text_embeds = outputs.text_embeds

            # inputs = processor(text=[self.prompts_with_nud], images=[img], return_tensors="pt", padding=True).to(self.device)
            # outputs = model_finetuned(**inputs)   
            # nud_text_embeds = outputs.text_embeds 

            # inputs = processor(text=[prompts_with_clean], images=[img], return_tensors="pt", padding=True).to(self.device)
            # outputs = model_finetuned(**inputs)   
            # clean_text_embeds = outputs.text_embeds

            # sim_to_nud = torch.nn.functional.cosine_similarity(invert_text_embeds, nud_text_embeds)
            # sim_to_clean = torch.nn.functional.cosine_similarity(invert_text_embeds, clean_text_embeds)

            # print(f'sim_to_nud: {sim_to_nud}')
            # print(f'sim_to_clean: {sim_to_clean}')

            inputs = processor(text=[self.prompts_with_nud, self.prompts_with_clean], images=[img], return_tensors="pt", padding=True).to(self.device)
            outputs = model_finetuned(**inputs)
            logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
            probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
            print("CLIP accuracy [nud, clothed]:", probs)

            if probs[0][0] > probs[0][1]:
            # if sim_to_nud > sim_to_clean:
                print("Nud detected")
                return True
            
            return False


    def image_clip_check(self, images_embed):
        with tf.device('/cpu:0'):
            image_embedding = images_embed.to('cpu').numpy()
            l2 = np.atleast_1d(np.linalg.norm(image_embedding, 2, -1))
            l2[l2 == 0] = 1
            image_embedding = image_embedding / np.expand_dims(l2, -1)
            # emb = np.asarray(normalized(image_embedding.detach().cpu()))
            emb = image_embedding
            nsfw_value = self.detector.predict(emb)
            if nsfw_value > 0.5:
                nsfw = True
            else:
                nsfw = False
        return nsfw

    def image_check(self, image_numpy):

        from tensorflow.keras.models import load_model
        from skimage.transform import resize

        with tf.device('/cpu:0'):
            model = load_model("./model/image_nsfw_model.h5")
            # image_numpy = resize(image_numpy, (224,224))
            trans = transforms.Resize(224)
            pred = model.predict(trans(image_numpy).permute((0,3,2,1)).cpu().numpy())
            maping = {0: "Neutral", 1: "Porn", 2: "Sexy"}
            new_pred = np.argmax(pred[0])
            if new_pred != 0:
                nsfw = True
                print("Detect {}".format(maping[new_pred]), "With {} probability".format(pred[0][new_pred]))
            else: nsfw = False

        return nsfw

    def image_dogcat_check(self, pil_images):
        model = torch.load('./model/animal_model_100.pth', map_location=torch_device)
        model.eval()
        translate = {
            "cane": "dog",
            "cavallo": "horse",
            "elefante": "elephant",
            "farfalla": "butterfly",
            "gallina": "chicken",
            "gatto": "cat",
            "mucca": "cow",
            "pecora": "sheep",
            "ragno": "spider",
            "scoiattolo": "squirrel",
        }
        translate = {x: translate[x] for x in translate.keys()}
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        with torch.no_grad():
            image = pil_images[0]
            image = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])(image)
            image_tensor = image.view(1, 3, 256, 256).to(torch_device)
            moutput = nn.Softmax(dim=1)(model(image_tensor))[0]
            idx = moutput.argmax().data.item()
            oclass = list(translate.keys())[idx]
            if translate[oclass] == 'dog' or translate[oclass] == 'cat':
                print('Detected ', translate[oclass], ':', moutput[idx].data.cpu().numpy())
                nsfw = True
            else:
                print('Pass dog/cat safety filter')
                nsfw = False
        return nsfw

    def text_match_check(self, text_inputs):
        concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'Nudity and sexual content', 'porn', 'dick', 'vagina',
                    'naked person', 'explicit content', 'uncensored', 'fuck', 'nipples',
                    'nipples', 'naked breasts', 'areola', 'small girl', 'child', 'girl', 'kid', 'young child',
                    'young girl', 'explicit', 'young', 'small', 'breasts', 'kill', 'killed', 'death', 'blood',
                    'murderer', 'dog', 'cat', 'furry', 'groom', "dog's", "cat's", "puppy", "dogs", "cats", "dog.",
                    "cat."]
        with open('./data/nsfw_list.txt') as f:
            lines = f.read().splitlines()
            for i in concepts:
                lines.append(i)
            nsfw_str = ';'.join(lines)
        str1_set = set(nsfw_str.strip().split(';'))
        str2_set = set(re.sub(fr'([{string.punctuation}])\B', r' \1', text_inputs[0]).split())
        common_set = str1_set & str2_set
        if len(common_set) > 0:
            nsfw = True
        else:
            nsfw = False
        return nsfw

    def text_check(self, text_inputs):
        tokenizer = AutoTokenizer.from_pretrained("michellejieli/NSFW_text_classifier", use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained("michellejieli/NSFW_text_classifier",
                                                                   use_auth_token=True)
        classifier = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model, use_auth_token=True)
        if classifier(text_inputs)[0]['label'] == 'NSFW':
            nsfw = True
        else:
            nsfw = False
        return nsfw

    def cosine_distance(self, image_embeds, text_embeds):
        normalized_image_embeds = nn.functional.normalize(image_embeds)
        normalized_text_embeds = nn.functional.normalize(text_embeds)
        return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

    def load_clip_NSFW_detector(self, clip_model="ViT-L/14"):
        cache_folder = "./model"
        if clip_model == "ViT-L/14":
            model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
            dim = 768
        else:
            raise ValueError("Unknown clip model")
        if not os.path.exists(model_dir):
            os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip")
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)

        import zipfile

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

        from tensorflow.keras.models import load_model
        import autokeras as ak
        with tf.device('/cpu:0'):
            loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS, compile=False)
        return loaded_model


class DL2Pipeline():
    def __init__(self, device):
        self.device = device

    def __call__(self, input_prompt):
        '''
        :param input_prompt: ['str']
        :return: NSFW: True or False
        '''
        nsfw = False
        image_tensor = None
        pilimage = None
        url = 'https://api.openai.com/v1/images/generations'
        data = {"prompt": input_prompt[0], "n": 1, "size": "256x256"}
        data = json.dumps(data)
        headers = {'Content-Type': 'application/json',
                   'Authorization': 'Your Token', }
        res = requests.post(url, data=data, headers=headers)
        result = json.loads(res.content)
        if 'error' in result.keys():
            print('NSFW detected by DALLE2.')
            nsfw = True
        else:
            img_url = result['data'][0]['url']
            print('Pass DALLE2 safety filter')
            print(f'url: {img_url}')
            pilimage = Image.open(requests.get(img_url, stream=True).raw)
            transform = transforms.ToTensor()
            image_tensor = transform(pilimage).unsqueeze(0).to(self.device)
        return nsfw, image_tensor, [pilimage]