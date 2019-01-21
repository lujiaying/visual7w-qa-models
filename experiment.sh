# Use jy-py2.7 virtualenv

# Sep 15, 11:50
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_train_top3.json" --output_json "data/qa_data_distract_train_top3.json" --output_h5 "data/qa_data_distract_train_top3.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 0 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_distract_train_top3.h5" -input_json "data/qa_data_distract_train_top3.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_train_top3.json"

# Sep 16, 09:20
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_train_dump_top3.json" --output_json "data/qa_data_distract_train_dump_top3.json" --output_h5 "data/qa_data_distract_train_dump_top3.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 0 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_distract_train_dump_top3.h5" -input_json "data/qa_data_distract_train_dump_top3.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_train_dump_top3.json"

# Sep 17, 14:57
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_word_emb.json" --output_json "data/qa_data_distract_word_emb.json" --output_h5 "data/qa_data_distract_word_emb.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 0 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_distract_word_emb.h5" -input_json "data/qa_data_distract_word_emb.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_word_emb.json"

# Sep 17, 15:59
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_add_top.json" --output_json "data/qa_data_distract_add_top.json" --output_h5 "data/qa_data_distract_add_top.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 0 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_distract_add_top.h5" -input_json "data/qa_data_distract_add_top.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_add_top.json"

# Sep 18, 17:45
# Generate detail for prediction
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 2 -batch_size 64 -mc_evaluation

# Sep 20, 14:29
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_inferSent.json" --output_json "data/qa_data_distract_inferSent.json" --output_h5 "data/qa_data_distract_inferSent.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 1 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_distract_inferSent.h5" -input_json "data/qa_data_distract_inferSent.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_inferSent.json"

# Sep 23, 16:49
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_SEA_rule_naive.json" --output_json "data/qa_data_SEA_rule_naive.json" --output_h5 "data/qa_data_SEA_rule_naive.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 1 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_SEA_rule_naive.h5" -input_json "data/qa_data_SEA_rule_naive.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_SEA_rule_naive.json"


# Oct 16, 15:49
th demo_generate_all_ans.lua -model checkpoints/model_visual7w_telling_gpu.t7 -gpuid 1
# ./results/telling_generation.txt store the basic generation result by pre-trained telling model
# Oct 17, 15:06
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_visual7w_baseline_g.json" --output_json "data/qa_data_visual7w_baseline_g.json" --output_h5 "data/qa_data_visual7w_baseline_g.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 1 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_visual7w_baseline_g.h5" -input_json "data/qa_data_visual7w_baseline_g.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_visual7w_baseline_g.json"

# Oct 28, 20:17, start RL without baseline training
th train_telling_policy_gradient.lua -load_model_from "checkpoints/model_visual7w_telling_gpu.t7" -suffix 1 -gpuid 2 -finetune_cnn_after -1 -batch_size 16

# Dec 3, 15:06, BoWClassifer non-rl baseline for distacting answer
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_nonRL.json" --output_json "data/qa_data_BoWClassifier_nonRL.json" --output_h5 "data/qa_data_BoWClassifier_nonRL.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 1 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_BoWClassifier_nonRL.h5" -input_json "data/qa_data_BoWClassifier_nonRL.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_nonRL.json"

# Dec 3, serving mc model as reward
~/luvit serve_mc_reward_model.lua

# Dec 4, 16:35 BoWClassifer rl for distacting answer
python generate_distracting_answers.py   # modify file_path in load_visual7w_baseline_candidate_dict()
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_RL_Nov03_9.json" --output_json "data/qa_data_BoWClassifier_RL_Nov03_9.json" --output_h5 "data/qa_data_BoWClassifier_RL_Nov03_9.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 2 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_BoWClassifier_RL_Nov03_9.h5" -input_json "data/qa_data_BoWClassifier_RL_Nov03_9.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_RL_Nov03_9.json"
# Dec 5, 11:35 BoWClassifer rl for distacting answer
python generate_distracting_answers.py   # modify file_path in load_visual7w_baseline_candidate_dict()
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_RL_Nov04_39.json" --output_json "data/qa_data_BoWClassifier_RL_Nov04_39.json" --output_h5 "data/qa_data_BoWClassifier_RL_Nov04_39.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 2 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_BoWClassifier_RL_Nov04_39.h5" -input_json "data/qa_data_BoWClassifier_RL_Nov04_39.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_RL_Nov04_39.json"

# Dec 6, 16:40
python generate_distracting_answers.py   # modify file_path in load_visual7w_baseline_candidate_dict()
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_RL_WithBaseline_Nov06_18.json" --output_json "data/qa_data_BoWClassifier_RL_WithBaseline_Nov06_18.json" --output_h5 "data/qa_data_BoWClassifier_RL_WithBaseline_Nov06_18.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 2 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_BoWClassifier_RL_WithBaseline_Nov06_18.h5" -input_json "data/qa_data_BoWClassifier_RL_WithBaseline_Nov06_18.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_BoWClassifier_RL_WithBaseline_Nov06_18.json"

# Dec 16, 20:07
python generate_distracting_answers.py   # modify file_path in load_visual7w_baseline_candidate_dict()

# Jan 3
python prepare_dataset.py --dataset_json "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_Seq2seq_Reinforce_Dec25.json" --output_json "data/qa_data_Seq2Seq_Reinforce_Dec25.json" --output_h5 "data/qa_data_Seq2Seq_Reinforce_Dec25.h5"
th eval_telling.lua -model model_visual7w_telling_gpu.t7 -gpuid 2 -batch_size 40 -mc_evaluation -input_h5 "data/qa_data_Seq2Seq_Reinforce_Dec25.h5" -input_json "data/qa_data_Seq2Seq_Reinforce_Dec25.json" -dataset_file "visual7w-toolkit/datasets/visual7w-telling/dataset_distract_Seq2seq_Reinforce_Dec25.json"

# Jan 21
python generate_distracting_answers.py   # modify file_path in load_visual7w_baseline_candidate_dict()
python prepare_dataset.py
