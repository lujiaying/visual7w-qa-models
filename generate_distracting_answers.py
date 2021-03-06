import json
import random
import time
import os
from collections import Counter
import h5py
import numpy as np
import tqdm
import string
from gensim.models import KeyedVectors
from generate_SEA_correct_answer import NUMBER_MAP


DIR_CUR = os.path.dirname(os.path.abspath(__file__))
DIR_DATA_SET_TELLING = '%s/visual7w-toolkit/datasets/visual7w-telling/' % (DIR_CUR)
PATH_DATA_SET_TELLING = '%s/dataset.json' % (DIR_DATA_SET_TELLING)
PATH_TOP_ANSWER = '%s/top_answers.json' % (DIR_DATA_SET_TELLING)

UNK_TOKEN = 'UNK'

def is_semantic_equivalent(correct_ans, candidate_ans):
    if isinstance(correct_ans, unicode):
        correct_ans = correct_ans.encode('utf8')
    if isinstance(candidate_ans, unicode):
        candidate_ans = candidate_ans.encode('utf8')
    correct_ans = correct_ans.lower().translate(None, string.punctuation).strip()
    candidate_ans = candidate_ans.lower().translate(None, string.punctuation).strip()

    # case 1: digits
    if correct_ans.isdigit() == True and correct_ans in NUMBER_MAP:
        correct_ans = NUMBER_MAP[correct_ans]
    if candidate_ans.isdigit() == True and candidate_ans in NUMBER_MAP:
        candidate_ans = NUMBER_MAP[candidate_ans]

    if correct_ans == candidate_ans:
        return True
    elif correct_ans in candidate_ans:
        return True
    else:
        return False

VISUAL7W_BASELINE_CANDIDATE_DICT = {}
def load_visual7w_baseline_candidate_dict():
    if VISUAL7W_BASELINE_CANDIDATE_DICT:
        return VISUAL7W_BASELINE_CANDIDATE_DICT

    #file_path = '%s/data/telling_generation.txt' % (DIR_CUR)
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/BoWClassifer-vanilla-40-sampled_answers.txt'  #BowClassifier, non-rl training
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/BoWClassifer-rl-Nov03-9-sampled_answers.txt'  #BowClassifier, rl first trial
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/BoWClassifer-rl-Nov04-39-sampled_answers.txt'  #BowClassifier, rl 39 epoches
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/BoWClassifer-rl_with_baseline-Nov06-18-sampled_answers.txt'  #BowClassifier, rl with baseline 18 epoches
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/Multilabel-baseline_Dec14-50-test_set.tsv'  # Multilabel baseline
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/BoWClassifer-vanilla-40-sampled_answers_Dec17_SSE.txt'   # BowClassifier, SSE filter
    #file_path = '/media/drive/Jiaying/attack-vqa-rl/outputs/BoWClassifer-vanilla-40-sampled_answers_Dec17_SSE.txt'   # BowClassifier, SSE filter
    file_path = '/media/drive/Jiaying/Visual-Distractor-Generation/data/baselines/adversarial_matching_distractors.tsv'  # Adversarial Matching
    with open(file_path) as fopen:
        for line in fopen:
            line_list = line.strip().split('\t')
            qa_id = int(line_list[1])
            golden_answer = line_list[3]
            candidates = line_list[-1].split('\001')
            candidates_processed = []
            for candidate in candidates:  #process candidates
                if candidate in golden_answer:
                    continue
                else:
                    candidates_processed.append(candidate)
            VISUAL7W_BASELINE_CANDIDATE_DICT[qa_id] = candidates_processed
    return VISUAL7W_BASELINE_CANDIDATE_DICT

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

def preprocess_inferSent_emb():
    """
    Convert sentence to int, produce w2v and id2sen files
    """
    inferSent_path = '%s/dataset.InferSent.vec' % (DIR_DATA_SET_TELLING)
    w2v_path = './data/inferSent.w2v.txt'
    id2sent_path = './data/inferSent.dict'
    dictionary = {}
    line_cnt = 0
    with open(inferSent_path) as fopen, open(w2v_path, 'w') as fwrite:
        fwrite.write('108626 4096\n')
        for line in fopen:
            sent, vec = line.strip().split('\t')
            dictionary[line_cnt] = sent
            fwrite.write('%s %s\n' % (line_cnt, vec))
            line_cnt += 1
    json.dump(dictionary, open(id2sent_path, 'w'))
    return 

def generate_cosine_matrix_inferSent():
    """
    Due to the scale of inferSent matric, the efficient way to find the most similar sentence is matrix multiplication

    Reference: https://stackoverflow.com/questions/41905029/create-cosine-similarity-matrix-numpy?rq=1
    """
    print('%s, start' % (time.ctime()))
    n_size = 108626 
    n_dimension = 4096

    emb_matrix = np.zeros((n_size, 4096))
    inferSent_path = '%s/dataset.InferSent.vec' % (DIR_DATA_SET_TELLING)
    with open(inferSent_path) as fopen:
        line_cnt = 0
        for line in fopen:
            sent, vec = line.strip().split('\t')
            vec_arr = [float(_) for _ in vec.split(' ')]
            emb_matrix[line_cnt] = vec_arr
            line_cnt += 1
    print('%s, matrix load DONE' % (time.ctime()))

    #d = np.dot(emb_matrix, emb_matrix.T)
    # dot memory not enough
    d = np.zeros((n_size, n_size))  # still too large
    for i in tqdm.tqdm(range(n_size)):
        for j in range(n_size):
            d[i][j] = np.dot(emb_matrix[i], emb_matrix[j])
    print('%s, dot DONE' % (time.ctime()))

    norm = (emb_matrix * emb_matrix).sum(1) ** .5
    print('%s, norm DONE' % (time.ctime()))

    sim_matrix = d / norm / norm.T
    print('%s, sim matrix DONE' % (time.ctime()))
    json.dump(sim_matrix.tolist(), open('data/inferSent.sim_matrix.json', 'w'), indent=2)
    return

def generate_similar_sentence_by_inferSent():
    print('%s, start' % (time.ctime()))
    wv_from_text = KeyedVectors.load_word2vec_format('./data/inferSent.w2v.txt', binary=False)
    id2sent = json.load(open('./data/inferSent.dict'))
    sent2id = dict([(id2sent[_], _) for _ in id2sent])
    print('%s, load DONE' % (time.ctime()))
    
    test_correct_ans_path = '%s/all_correct_answer' % (DIR_DATA_SET_TELLING)
    similar_sent_path = '%s/similar_sent.inferSent.txt' % (DIR_DATA_SET_TELLING)
    with open(test_correct_ans_path) as fopen, open(similar_sent_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            ans_id = sent2id[line_list[0].lower()]
            res = wv_from_text.similar_by_vector(str(ans_id), 5)
            res_detail = [(id2sent[_[0]], _[1]) for _ in res]
            fwrite.write('%s\t%s\t%s\n' % (line_list[0], line_list[1], json.dumps(res_detail)))
    print('%s, generate DONE' % (time.ctime()))
    return

def generate_all_test_correct_answer():
    # Load original dataset
    dataset = json.load(open(PATH_DATA_SET_TELLING, 'r'))
    all_correct_ans = {}

    for image_idx in tqdm.tqdm(range(len(dataset['images']))):
        image = dataset['images'][image_idx]
        img_split = image['split']
        if img_split != 'test':
            continue
        for qa_pair_idx in range(len(image['qa_pairs'])):
            qa_pair = image['qa_pairs'][qa_pair_idx]
            qa_type = qa_pair['type']
            correct_ans = qa_pair['answer']
            if correct_ans not in all_correct_ans:
                all_correct_ans[correct_ans] = 0
            all_correct_ans[correct_ans] += 1

    with open('%s/all_correct_answer' % (DIR_DATA_SET_TELLING), 'w') as fwrite:
        for k, v in sorted(all_correct_ans.items(), key=lambda _:_[1], reverse=True):
            fwrite.write('%s\t%s\n' % (k, v))
    return

def get_inferSent_embbedings():
    similar_sent_dict = {}
    with open('%s/similar_sent.inferSent.txt' % (DIR_DATA_SET_TELLING)) as fopen:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            correct_ans = line_list[0]
            similar_snet_list = json.loads(line_list[2])
            similar_sent_dict[correct_ans] = similar_snet_list
    return similar_sent_dict

def generate_distracting_answer_inferSent(correct_ans, similar_sent_dict):
    """
    Args:
        correct_ans: str
        similar_sent_dict: dict, {sent:[(sent_s, sim_s), ()]}
    """
    res = []
    if isinstance(correct_ans, unicode):
        correct_ans = correct_ans.encode('utf8')
    correct_ans_processed = correct_ans.lower().translate(None, string.punctuation).strip()
    for sent, score in similar_sent_dict[correct_ans]:
        if isinstance(sent, unicode):
            sent = sent.encode('utf8')
        sent_processed = sent.lower().translate(None, string.punctuation).strip()
        if correct_ans_processed == sent_processed:
            continue
        else:
            res.append(sent)
            if len(res) == 3:
                break
    return res

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

def get_distracting_answer_by_visual7w_generation(qa_id, correct_answer, candidate_dict, res_cnt=3):
    """
    Args:
        qa_id: int
        correct_answer: str
        candidate_dict: dict of (id, candidate), generated by visual7w baseline model
    """
    res = []
    if qa_id not in candidate_dict:
        print('%s not in candidate_dict' % (qa_id))
        return res
    for candidate in candidate_dict[qa_id]:
        if is_semantic_equivalent(correct_answer, candidate):
            continue
        if candidate not in res:
            res.append(candidate)
        if len(res) >= res_cnt:
            break
    return res

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

    # load inferSent emb
    similar_sent_dict = get_inferSent_embbedings()
    error_multiple_choices = 0

    # Generate distracting answers
    for image_idx in tqdm.tqdm(range(len(dataset['images']))):
        image = dataset['images'][image_idx]
        img_split = image['split']
        if img_split != 'test':
            continue
        for qa_pair_idx in range(len(image['qa_pairs'])):
            qa_pair = image['qa_pairs'][qa_pair_idx]
            qa_type = qa_pair['type']
            qa_id = int(qa_pair['qa_id'])
            correct_ans = qa_pair['answer']
            #qa_pair['multiple_choices'] = generate_distracting_answer_by_top(qa_type, top_answer_train)
            #qa_pair['multiple_choices'] = generate_distracting_answer_dump(qa_type, top_answer_train)
            #qa_pair['multiple_choices'] = [_[0] for _ in get_similar_word_by_emb(wv_from_text, correct_ans)]
            #qa_pair['multiple_choices'] = generate_distracting_answer_add_top(qa_type, top_answer_train, correct_ans)
            #qa_pair['multiple_choices'] = generate_distracting_answer_inferSent(correct_ans, similar_sent_dict)
            distracting_candidates = get_distracting_answer_by_visual7w_generation(qa_id, correct_ans, 
                    load_visual7w_baseline_candidate_dict())
            qa_pair['multiple_choices'][:len(distracting_candidates)] = distracting_candidates
            if len(distracting_candidates) != 3:
                error_multiple_choices += 1

    #distract_ans_path = '%s/dataset_distract_train_top3.json' % (DIR_DATA_SET_TELLING)   # use top3 of training data to replace
    #distract_ans_path = '%s/dataset_distract_train_dump_top3.json' % (DIR_DATA_SET_TELLING) # use other type top3 'train split' to replace
    #distract_ans_path = '%s/dataset_distract_word_emb.json' % (DIR_DATA_SET_TELLING) # use embedding top3 similarity 
    #distract_ans_path = '%s/dataset_distract_add_top.json' % (DIR_DATA_SET_TELLING) # add phrase of top3 answer 
    #distract_ans_path = '%s/dataset_distract_inferSent.json' % (DIR_DATA_SET_TELLING) # inferSent for similar sent embed
    #distract_ans_path = '%s/dataset_distract_visual7w_baseline_g.json' % (DIR_DATA_SET_TELLING) # visual7w generation mode
    #distract_ans_path = '%s/dataset_distract_BoWClassifier_nonRL.json' % (DIR_DATA_SET_TELLING) # BoWClassifier non-rl 
    #distract_ans_path = '%s/dataset_distract_BoWClassifier_RL_Nov03_9.json' % (DIR_DATA_SET_TELLING) # BoWClassifier rl, first trial
    #distract_ans_path = '%s/dataset_distract_BoWClassifier_RL_Nov04_39.json' % (DIR_DATA_SET_TELLING) # BoWClassifier rl, 39 epoches
    #distract_ans_path = '%s/dataset_distract_BoWClassifier_RL_WithBaseline_Nov06_18.json' % (DIR_DATA_SET_TELLING) # BoWClassifier rl with baseline ,18 epoches
    #distract_ans_path = '%s/dataset_distract_Multilabel-baseline_Dec14-50-test_set.json' % (DIR_DATA_SET_TELLING) # Multilabel baseline
    #distract_ans_path = '%s/dataset_distract_BoWClassifier_40-sampled_Dec17_SSE_test_set.json' % (DIR_DATA_SET_TELLING) # BowClassifier SSE Filter
    distract_ans_path = '%s/dataset_distract_Adversarial_Matching_Jan18.json' % (DIR_DATA_SET_TELLING) # BowClassifier SSE Filter
    print('error_multiple_choices:%s' % (error_multiple_choices))
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

    # sentence encoding
    # preprocess_inferSent_emb()
    # generate_cosine_matrix_inferSent()
    #generate_all_test_correct_answer()
    #generate_similar_sentence_by_inferSent()
