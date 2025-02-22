import random



def load_data(path):
    prompt_list = []
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    for i in lines:
        prompt_list.append([i])
    return prompt_list


new_list = random.sample(load_data("data/nsfw_list.txt"), 200)

with open('data/nsfw_list_sample.txt', 'w') as fp:
    for item in new_list:
        # write each item on a new line
        fp.write("%s\n" % item[0])
    print('Done')