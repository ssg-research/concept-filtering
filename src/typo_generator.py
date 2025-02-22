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

import numpy as np
 
def generate_typo(string, typos=1):
        """Perturbation functions, swaps random characters with their neighbors
        Could also randomly duplicate characters, remove characters, or replace o's with zero,
        or l's and I's with 1's

        Parameters
        ----------
        string : str
            input string
        typos : int
            number of typos to add

        Returns
        -------
        list(string)
            perturbed strings

        """
        typo_choice = np.random.randint(1,4)
        to_ret = string

        if typo_choice == 1:
            string = list(string)
            swaps = np.random.choice(len(string) - 1, typos)
            for swap in swaps:
                tmp = string[swap]
                string[swap] = string[swap + 1]
                string[swap + 1] = tmp
            to_ret = ''.join(string)
        
        if typo_choice == 2:
            string = list(string)
            dups = np.random.choice(len(string) - 1, typos)
            for dup in dups:
                # Insert at position dup the character at position dup+1
                string.insert(dup, string[dup+1])
            to_ret = ''.join(string)

        if typo_choice == 3:
            string = list(string)
            removes = np.random.choice(len(string) - 1, typos)
            for rem in removes:
                # Insert at position dup the character at position dup+1
                string.pop(rem)
            to_ret = ''.join(string)
        
        # Replace o with 0?
        replace_o_with_0 = np.random.randint(3)
        if replace_o_with_0 == 2:
            to_ret = to_ret.replace('o', '0').replace('O','0')
        
        replace_l_with_1 = np.random.randint(3)
        if replace_l_with_1 == 2:
            to_ret = to_ret.replace('l', '1').replace('I','1')