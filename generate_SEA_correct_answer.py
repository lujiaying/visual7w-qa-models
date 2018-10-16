"""
Generate Senmantic Equivalent Adversarial Correct Answer
"""

import json
import random
import time
import os
from collections import Counter
import h5py
import numpy as np
import tqdm
import string

DIR_CUR = os.path.dirname(os.path.abspath(__file__))
DIR_DATA_SET_TELLING = '%s/visual7w-toolkit/datasets/visual7w-telling/' % (DIR_CUR)
PATH_DATA_SET_TELLING = '%s/dataset.json' % (DIR_DATA_SET_TELLING)
PATH_TOP_ANSWER = '%s/top_answers.json' % (DIR_DATA_SET_TELLING)

UNK_TOKEN = 'UNK'
NUMBER_MAP = {
        'zero': 0,     '0': 'zero',
        "one": '1',    '1': "one",
        'two': '2',    '2': 'two',
        'three': '3',  '3': 'three',
        'four': '4',   '4': 'four',
        'five': '5',   '5': 'five',
        'six': '6',    '6': 'six',
        'seven': '7',  '7': 'seven',
        'eight': '8',  '8': 'eight',
        'nine': '9',   '9': 'nine',
        'ten': '10',   '10': 'ten',
        }
COLOR_LIST = ["white", 'black', 'green', 'blue', 'red', 'brown', 'yellow', 'gray', 'orange', 'grey', 'pink', 'gold', 'sliver', 'black and white']
NEGATIVE_MAP = {'no one': 'nobody', 'nobody': 'no one'}
WEATHER_LIST = ['sunny', 'rainy', 'stormy', 'cloudy', 'snowy', 'raining', 'cold', 'hot', 'windy']

def generate_by_rules(correct_ans):
    """
    R1: Two -> 2, 3 -> Three       (NUMBER -> Digit, vice versa)
    R2: One/a man -> The man  (One/a <NOUN> -> The <NOUN>, vice versa)
    R3: Grey. -> It's grey.            (COLOR -> It's COLOR.)
    R4: In front of XX. -> Ahead of XX.   (POSITION -> POSITION)
    R5: No one. -> Nobody. Nobody. -> No one.
    R6: Sunny. -> It's sunny.

    Returns: 
        rephrase_ans: str
    """
    if isinstance(correct_ans, unicode):
        correct_ans = correct_ans.encode('utf8')
    processed_correct_ans = correct_ans.lower().translate(None, string.punctuation).strip()

    # Rule 1
    if processed_correct_ans in NUMBER_MAP:
        rephrase_ans = NUMBER_MAP[processed_correct_ans]
        return rephrase_ans

    # Rule 2: temporary not add

    # Rule 3
    if processed_correct_ans in COLOR_LIST:
        rephrase_ans = "It's %s." % (processed_correct_ans)
        return rephrase_ans

    # Rule 4: hit too less

    # Rule 5
    if processed_correct_ans in NEGATIVE_MAP:
        rephrase_ans = NEGATIVE_MAP[processed_correct_ans]
        return rephrase_ans

    # Rule 6
    if processed_correct_ans in WEATHER_LIST:
        rephrase_ans = "It's %s" % (processed_correct_ans)
        return rephrase_ans
    
    return correct_ans

def main():
    # Load original dataset
    dataset = json.load(open(PATH_DATA_SET_TELLING, 'r'))

    # Generate distracting answers
    hit_num = 0
    for image_idx in tqdm.tqdm(range(len(dataset['images']))):
        image = dataset['images'][image_idx]
        img_split = image['split']
        if img_split != 'test':
            continue
        for qa_pair_idx in range(len(image['qa_pairs'])):
            qa_pair = image['qa_pairs'][qa_pair_idx]
            qa_type = qa_pair['type']
            correct_ans = qa_pair['answer']
            rephrase_ans = generate_by_rules(correct_ans)
            if rephrase_ans != correct_ans:
                hit_num += 1
                qa_pair['answer'] = rephrase_ans

    distract_ans_path = '%s/dataset_SEA_rule_naive.json' % (DIR_DATA_SET_TELLING) # Only Rule 1,3,5,6
    print('hit rephrase num:%d' % (hit_num))
    with open(distract_ans_path, 'w') as fw:
        json.dump(dataset, fw, indent=2)

if __name__ == '__main__':
    main()
