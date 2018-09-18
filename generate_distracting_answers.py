import json
import random
import os
from collections import Counter
import h5py
import numpy as np
import tqdm
import string
from gensim.models import KeyedVectors


DIR_CUR = os.path.dirname(os.path.abspath(__file__))
DIR_DATA_SET_TELLING = '%s/visual7w-toolkit/datasets/visual7w-telling/' % (DIR_CUR)
PATH_DATA_SET_TELLING = '%s/dataset.json' % (DIR_DATA_SET_TELLING)
PATH_TOP_ANSWER = '%s/top_answers.json' % (DIR_DATA_SET_TELLING)

UNK_TOKEN = 'UNK'

def generate_gensim_word2vec():
    emb_matrix_para_path = '%s/data/telling_gpu_word_emb_parameters.txt' % (DIR_CUR)
    vocab_path = '%s/data/qa_data.json' % (DIR_CUR)
    output_path = '%s/data/w2v_telling_gpu.txt' % (DIR_CUR)

    vocab_dict = json.load(open(vocab_path))
    with open(emb_matrix_para_path, 'r') as fo, open(output_path, 'w') as fw:
        fw.write('3006 512\n')
        for idx, line in enumerate(fo):
            if idx+1 > 3006:
                break
            word = vocab_dict['ix_to_word'][str(idx+1)]
            fw.write('%s %s' % (word, line))

def get_similar_word_by_emb(w2v, answer, topk=3):
    """
    Args:
        w2v: KeyedVectors instance
        answer: string
        topk: int
    Returns:
        sim_words: list of (word, similarity)
    """
    ans_emb = None
    if isinstance(answer, unicode):
        answer = answer.encode('utf8')
    processed_answer = answer.lower().translate(None, string.punctuation).strip().split()
    if len(processed_answer) == 1:
        token = processed_answer[0] if processed_answer[0] in w2v else UNK_TOKEN
        return w2v.similar_by_word(token, topn=topk)
    for w in processed_answer:
        token = w if w in w2v else UNK_TOKEN
        if isinstance(ans_emb, type(None)):
            ans_emb = w2v[token].copy()
        else:
            ans_emb += w2v[token]
    return w2v.similar_by_vector(ans_emb, topn=topk)

def generate_distracting_answer_by_top(question_type, top_answers_split):
    """
    Use top 3 answer of specific type to replace
    Args:
        question_type: string
        top_answers_split: dict, keys are question_type, values are list of (top_answers, frequency)
    """
    return [_[0] for _ in top_answers_split[question_type][:3]]

def generate_distracting_answer_dump(question_type, top_answers_split):
    """
    Use other type top 3 answer to replace
    """
    question_types = top_answers_split.keys()
    target_q_type = ''
    while not target_q_type:
        random_q_type = random.choice(question_types)
        if random_q_type != question_type:
            target_q_type = random_q_type
    return [_[0] for _ in top_answers_split[target_q_type][:3]]

def generate_distracting_answer_add_top(question_type, top_answers_split, correct_answer):
    """
    Add noise phrase before, after, before&after correct_answer
    """
    top_distracting_ans_list = generate_distracting_answer_by_top(question_type, top_answers_split)

    chosen_idx_list = random.sample(range(len(top_distracting_ans_list)), 2)
    primary_phrase = top_distracting_ans_list[chosen_idx_list[0]]
    secondary_phrase = top_distracting_ans_list[chosen_idx_list[1]] 

    return [
            primary_phrase + ' ' + correct_answer, 
            correct_answer + ' ' + primary_phrase, 
            primary_phrase + ' ' + correct_answer + ' ' + secondary_phrase,
           ]

def generate_distracting_answer_by_similar_entity():
    """
    Args:
        original_answer:
        word_embeddings:
    """
    pass

def get_question_top_answer(split):
    """
    Args:
        split: string, train | val | test | all
    """
    topk = 50

    if os.path.exists(PATH_TOP_ANSWER):
        with open(PATH_TOP_ANSWER, 'r') as fo:
            result = json.load(fo)
    else:
        result = {'train':{}, 'val':{}, 'test':{}, 'all':{}}
        answer_cnt = {'train':{}, 'val':{}, 'test':{}, 'all':{}}
        with open(PATH_DATA_SET_TELLING, 'r') as fo, open(PATH_TOP_ANSWER, 'w') as fw:
            dataset = json.load(fo)
            for image in tqdm.tqdm(dataset['images']):
                img_split = image['split']
                for qa_pair in image['qa_pairs']:
                    qa_type = qa_pair['type']
                    answer = qa_pair['answer']
                    if qa_type not in answer_cnt[img_split]:
                        answer_cnt[img_split][qa_type] = Counter()
                    if qa_type not in answer_cnt['all']:
                        answer_cnt['all'][qa_type] = Counter()
                    answer_cnt[img_split][qa_type][answer] += 1
                    answer_cnt['all'][qa_type][answer] += 1
            for split in answer_cnt:
                for qa_type in answer_cnt[split]:
                    result[split][qa_type] = answer_cnt[split][qa_type].most_common(topk)
            json.dump(result, fw, indent=4)
    return result[split]

def main():
    # Load original dataset
    dataset = json.load(open(PATH_DATA_SET_TELLING, 'r'))
    # Load top answer
    top_answer_train = get_question_top_answer('train')
    # Load word embeddings
    wv_from_text = KeyedVectors.load_word2vec_format('./data/w2v_telling_gpu.txt', binary=False)

    # Generate distracting answers
    for image_idx in tqdm.tqdm(range(len(dataset['images']))):
        image = dataset['images'][image_idx]
        img_split = image['split']
        if img_split != 'test':
            continue
        for qa_pair_idx in range(len(image['qa_pairs'])):
            qa_pair = image['qa_pairs'][qa_pair_idx]
            qa_type = qa_pair['type']
            correct_ans = qa_pair['answer']
            #qa_pair['multiple_choices'] = generate_distracting_answer_by_top(qa_type, top_answer_train)
            #qa_pair['multiple_choices'] = generate_distracting_answer_dump(qa_type, top_answer_train)
            #qa_pair['multiple_choices'] = [_[0] for _ in get_similar_word_by_emb(wv_from_text, correct_ans)]
            qa_pair['multiple_choices'] = generate_distracting_answer_add_top(qa_type, top_answer_train, correct_ans)

    #distract_ans_path = '%s/dataset_distract_train_top3.json' % (DIR_DATA_SET_TELLING)   # use top3 of training data to replace
    #distract_ans_path = '%s/dataset_distract_train_dump_top3.json' % (DIR_DATA_SET_TELLING) # use other type top3 'train split' to replace
    #distract_ans_path = '%s/dataset_distract_word_emb.json' % (DIR_DATA_SET_TELLING) # use embedding top3 similarity 
    distract_ans_path = '%s/dataset_distract_add_top.json' % (DIR_DATA_SET_TELLING) # add phrase of top3 answer 
    with open(distract_ans_path, 'w') as fw:
        json.dump(dataset, fw, indent=2)

if __name__ == '__main__':
    #get_question_top_answer('all')
    main()

    """
    #generate_gensim_word2vec()
    wv_from_text = KeyedVectors.load_word2vec_format('./data/w2v_telling_gpu.txt', binary=False)

    ans = 'To tie up the boats.'
    res = get_similar_word_by_emb(wv_from_text, ans)
    print(ans, res)

    ans = 'in the sea'
    res = get_similar_word_by_emb(wv_from_text, ans)
    print(ans, res)

    ans = 'One.'
    res = get_similar_word_by_emb(wv_from_text, ans)
    print(ans, res)

    ans = 'Gulls.'
    res = get_similar_word_by_emb(wv_from_text, ans)
    print(ans, res)
    """
