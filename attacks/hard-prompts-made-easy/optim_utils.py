import random
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from statistics import mean
import copy
import json
from typing import Any, Mapping

import open_clip

import torch

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def contrastive_loss(logits):
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity):
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_target_feature(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    unfed = None
    if target_images is not None:
        with torch.no_grad():
            curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = torch.concatenate(curr_images).to(device)
            unfed =  curr_images
            print(f'num images: {len(curr_images)}')
            all_target_features = model.encode_image(curr_images)
    else:
        texts = tokenizer_funct(target_prompts).to(device)
        unfed = texts
        all_target_features = model.encode_text(texts)

    return all_target_features, unfed


def initialize_prompt(tokenizer, token_embedding, args, device):
    prompt_len = args.prompt_len

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    print(f'padded_template_text: {padded_template_text}')
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    
    return prompt_embeds, dummy_embeds, dummy_ids


def get_nud_clean_prompts(experiment):
    if experiment == 'nudity':
        prompts_with_nud = f'nude'
        prompts_with_clean = f"clean"

    elif experiment == 'grumpy':
        prompts_with_nud = 'grumpy cat'
        prompts_with_clean = 'cat'

    elif experiment == 'vangogh':
        prompts_with_nud = 'in the style of van gogh'
        prompts_with_clean = 'in standard style'

    elif experiment == 'monet':
        prompts_with_nud = f'in the style of Monet'
        prompts_with_clean = 'in standard style'

    elif  experiment == 'dali':
        prompts_with_nud = f'in the style of Dali'
        prompts_with_clean = 'in standard style'

    elif experiment == 'gregrut':
        prompts_with_nud = f'in the style of Greg Rutowski'
        prompts_with_clean = 'in standard style'

    elif experiment == 'r2d2':
        prompts_with_nud = 'r2d2'
        prompts_with_clean = 'robot'

    elif experiment == 'marvel':
        prompts_with_nud = 'captain marvel'
        prompts_with_clean = 'female superhero'

    elif experiment == 'nemo':
        prompts_with_nud = 'nemo'
        prompts_with_clean = 'fish'

    elif experiment == 'musk':
        prompts_with_nud = 'Elon Musk'
        prompts_with_clean = 'man'

    elif experiment == 'pitt':
        prompts_with_nud = 'Brad Pitt'
        prompts_with_clean = 'man'

    elif experiment == 'swift':
        prompts_with_nud = 'Taylor Swift'
        prompts_with_clean = 'woman'
    
    elif experiment == 'jolie':
        prompts_with_nud = 'Angelina Jolie'
        prompts_with_clean = 'woman'
    
    elif experiment == 'snoopy':
        prompts_with_nud = 'Snoopy'
        prompts_with_clean = 'dog'
    
    elif experiment == 'violence':
        prompts_with_nud = 'violence'
        prompts_with_clean = 'peaceful'
    
    return prompts_with_clean, prompts_with_nud


def optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device, unfed, tokenizer_funct):
    opt_iters = args.iter
    lr = args.lr
    weight_decay = args.weight_decay
    print_step = args.print_step
    batch_size = args.batch_size
    print_new_best = getattr(args, 'print_new_best', False)
    set_random_seed(seed=args.seed)
    experiment = args.experiment
    print(f'seed: {args.seed}')
    print(f'experiment: {experiment}')
    '''
    prompts_with_clean, prompts_with_nud = get_nud_clean_prompts(experiment)
    prompts_with_nud_tkn = tokenizer_funct(prompts_with_nud).to(device)
    prompts_with_nud_features = model.encode_text(prompts_with_nud_tkn)
    prompts_with_clean_tkn = tokenizer_funct(prompts_with_clean).to(device)
    prompts_with_clean_features = model.encode_text(prompts_with_clean_tkn)
    '''

    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_sim = -1000 * args.loss_weight
    best_text = ""
    # ourclip = False
    # model_2, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14') # USE OUR TEXT_MODEL INSTEAD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model_2 = model
    # if args.ourclip:
    #     print('Using our clip')
    #     # # DISABLING THE LOADING OF THE CLIP' MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     # model_2, _, preprocess_oc = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device) # This is the hf-to-pt version of model_finetune, used in pez
    #     # ckpt_oc = torch.load(f'attacks/hard-prompts-made-easy/ViT-L-14-openai-dec17_new/open_clip_pytorch_model.bin')
    #     # # ckpt_oc = torch.load(f'attacks/hard-prompts-made-easy/ViT-L-14-openai/open_clip_pytorch_model.bin')
    #     # model_2.load_state_dict(ckpt_oc)

    #     # nud model
    #     if experiment == 'nudity':
    #         ckpt_oc = torch.load(f'attacks/hard-prompts-made-easy/ViT-L-14-openai-dec17_new/open_clip_pytorch_model.bin')
    #         # ckpt_oc = torch.load(f'attacks/hard-prompts-made-easy/ViT-L-14-openai/open_clip_pytorch_model.bin')
    #     elif experiment == 'violence':
    #         ckpt_oc = torch.load(f'attacks/hard-prompts-made-easy/ViT-L-14-openai-violence_ourclip/open_clip_pytorch_model.bin')
    #     model.load_state_dict(ckpt_oc)
    


    # tokenizer_2 = open_clip.get_tokenizer('ViT-L-14')

    # loss_cos_emb = torch.nn.CosineEmbeddingLoss()
    # loss_img = torch.nn.CrossEntropyLoss()
    # loss_txt = torch.nn.CrossEntropyLoss()
    # logit_scale = model_2.logit_scale.exp()

    for step in range(opt_iters):
        # randomly sample sample images and get features
        if batch_size is None:
            target_features = all_target_features
        else:
            curr_indx = torch.randperm(len(all_target_features))
            target_features = all_target_features[curr_indx][0:batch_size]
            
        universal_target_features = all_target_features
        
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, print_hits=False)

        # get cosine similarity score with all target features
        with torch.no_grad():
            # padded_embeds = copy.deepcopy(dummy_embeds)
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, universal_target_features)
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()
        
        # tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True
        
        # padding
        # padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)
        
        logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, target_features)
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        loss = loss * args.loss_weight
        '''
        logits_per_image_nud, _ = model.forward_text_embedding_text_image(prompts_with_nud_features, target_features)
        logits_per_image_clean, _ = model.forward_text_embedding_text_image(prompts_with_clean_features, target_features)
        loss = loss + 0.99*(logits_per_image_clean.mean() - logits_per_image_nud.mean())
        '''
        
        
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
        
        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        decoded_text = decode_ids(nn_indices, tokenizer)[best_indx]
        if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
            per_step_message = f"step: {step}, lr: {curr_lr}"
            if not print_new_best:
                per_step_message = f"\n{per_step_message}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}"
            print(per_step_message)

        if best_sim * args.loss_weight < universal_cosim_score * args.loss_weight:
            best_sim = universal_cosim_score
            best_text = decoded_text
            if print_new_best:
                print(f"new best cosine sim: {best_sim}")
                print(f"new best prompt: {best_text}")


    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text


def optimize_prompt(model, preprocess, args, device, target_images=None, target_prompts=None, all_target_features = None):
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.get_tokenizer(args.clip_model)
    unfed = None

    if all_target_features is None:
    # get target features
        all_target_features, unfed = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images, target_prompts=target_prompts)

    # optimize prompt
    learned_prompt = optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device, unfed, tokenizer_funct)

    return learned_prompt
    

def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        ori_batch = torch.concatenate(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        gen_batch = torch.concatenate(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()