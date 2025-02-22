# Authors: Anudeep Das, Vasisht Duddu, Rui Zhang, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from clip_benchmark.datasets.builder import build_dataset
import pandas as pd
import os

root_path = "coco-dataset/" # set this to smth meaningful


def load_and_save(built_ds, split):
    coco = built_ds.coco
    imgs = coco.loadImgs(coco.getImgIds())
    future_df = {"filepath":[], "title":[]}
    for img in imgs:
        caps = coco.imgToAnns[img["id"]]
        for cap in caps:
            future_df["filepath"].append(img["file_name"])
            future_df["title"].append(cap["caption"])
    pd.DataFrame.from_dict(future_df).to_csv(
    os.path.join(root_path, f"{split}2014.csv"), index=False, sep="\t"
)

train_ds = build_dataset("mscoco_captions", root=root_path, split="train") # this downloads the dataset if it is not there already
test_ds = build_dataset("mscoco_captions", root=root_path, split="test")

load_and_save(train_ds, split='train')
load_and_save(test_ds, split='test')