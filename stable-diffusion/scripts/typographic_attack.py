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

from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default = 'nudity')
parser.add_argument('--suffix', type=str, default=1)
parser.add_argument('--image_path', type=str)
parser.add_argument('--prompts_with_nud', type=str, default='nude')
parser.add_argument('--prompts_with_clean', type=str, default='clean')
args = parser.parse_args()

# Specify the directory containing your images
input_directory = args.image_path

# Specify the directory where you want to save the modified images
output_directory = os.path.join('.', f'{args.experiment}_typographic_{args.suffix}')

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
file_list = os.listdir(input_directory)

# Loop through each file
for file_name in file_list:
    # Construct the full path of the input file
    input_path = os.path.join(input_directory, file_name)

    # Load the image
    image = Image.open(input_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the rectangle parameters
    rect_width = image.width
    rect_height = 40  # Adjust this value based on your preference
    # rect_color = 'white'

    # Draw the white rectangle at the bottom
    # draw.rectangle([(0, image.height - rect_height), (rect_width, image.height)], fill=rect_color)

    # Specify the font and text color
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, 128)
    text_color = 'blue'

    # Specify the text
    text = args.prompts_with_clean

    # Calculate the position to center the text in the rectangle
    text_position = ((image.width - draw.textbbox(xy = (image.height - 300, 0 + 400), text=text, font=font)[0]) // 2, image.height - 8*rect_height + 10)

    # Draw the text on the image
    draw.text(text_position, text, font=font, fill=text_color)

    # Construct the full path of the output file
    output_path = os.path.join(output_directory, file_name)

    # Save the modified image to the output directory
    image.save(output_path)
    print(f'Image saved at {output_path}')

# # Optionally, you can also close the image after processing all files
# image.close()
