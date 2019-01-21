require 'torch'
require 'nn'
require 'nngraph' require 'loadcaffe'
require 'image'
require 'math'

require 'misc.QADatasetLoader'
require 'modules.QAAttentionModel'
require 'modules.QACriterion'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

local rl_env = {}

local rnn = nil
local cnn = nil
local crit = nil
local seq_length = 20
local vocab_size = 3007
local word_to_ix = {}
local gpu_mode = false

function rl_env.init(gpuid)
  if rnn == nil then
    print(string.format('Init reinforcement learning environment, gpuid=%s', gpuid))

    --[[
    local loader = QADatasetLoader{h5_file='data/qa_data.h5', json_file='data/qa_data.json',dataset_file='visual7w-toolkit/datasets/visual7w-telling/dataset.json'}
    seq_length = loader.seq_length
    vocab_size = loader.vocab_size
    print(seq_length)
    print(vocab_size)
    loader = nil
    collectgarbage() -- free some memory
    --]]

    gpu_mode = gpuid >= 0
    if gpu_mode then
      require 'cutorch'
      require 'cunn'
      require 'cudnn'
      cutorch.manualSeed(1234)
      cutorch.setDevice(gpuid + 1) -- note +1 because lua is 1-indexed
    end

    local checkpoint = torch.load('checkpoints/model_visual7w_telling_gpu.t7')
    local vocab = checkpoint.vocab
    for i, w in pairs(vocab) do
      word_to_ix[w] = i
    end
    local modules = checkpoint.modules
    cnn = modules.cnn
    rnn = modules.rnn
    modules = nil
    collectgarbage() -- free some memory
    if gpu_mode then cnn:cuda() end
    rnn:createClones()
    if gpu_mode then rnn:cuda() end
    cnn:evaluate()
    rnn:evaluate()

    crit = nn.QACriterion()
    if gpu_mode then crit:cuda() end
  end

  print(string.format('Done init reinforcement learning environment, gpuid=%s', gpuid))
  return rnn, cnn, crit
end

function rl_env.get_features(image_ids, questions, answers)
  local batch_size = #image_ids
  local q_lens = utils.get_sequences_token_cnt(questions)
  local a_lens = utils.get_sequences_token_cnt(answers)

  local MAX_Q_LEN = 14

  local data = {}
  data.images = torch.Tensor(batch_size, 3, 224, 224)
  data.question_lengths = torch.LongTensor(batch_size)
  data.labels = torch.LongTensor(batch_size, seq_length)
  for i = 1, batch_size do
    local image_path = string.format('images/v7w_%s.jpg', image_ids[i])
    local img = image.load(image_path, 3)
    img = image.scale(img, 224, 224, 'bicubic'):mul(255)
    data.images[i] = img

    local seq = torch.LongTensor(seq_length):zero()
    local question = string.lower(questions[i]):gsub('%p', '')
    --local q_len = q_lens[i]
    local q_len = math.min(q_lens[i], MAX_Q_LEN)
    data.question_lengths[i] = q_len + 1  -- with <START> token
    local question_label = torch.LongTensor(q_len):zero()
    local token_idx = 1
    for token in question:gmatch('%w+') do
      local word_ix = word_to_ix[token]
      if word_ix == nil then word_ix = word_to_ix['UNK'] end
      question_label[token_idx] = word_ix
      token_idx = token_idx + 1
      if token_idx > q_len then break end
    end
    seq[{{1,q_len}}] = question_label
    seq[q_len+1] = vocab_size

    local answer = string.lower(answers[i]):gsub('%p', '')
    local MAX_A_LEN = seq_length - 1 - q_len
    local a_len = math.min(a_lens[i], MAX_A_LEN)
    local ans_label = torch.LongTensor(a_len):zero()
    token_idx = 1
    for token in answer:gmatch('%w+') do
      local word_ix = word_to_ix[token]
      if word_ix == nil then word_ix = word_to_ix['UNK'] end
      ans_label[token_idx] = word_ix
      token_idx = token_idx + 1
      if token_idx > a_len then break end
    end
    seq[{{q_len+2, q_len+a_len+1}}] = ans_label
    data.labels[i] = seq
  end

  data.images = net_utils.prepro(data.images, false, gpu_mode)
  data.labels = data.labels:transpose(1,2):contiguous()
  data.question_lengths = data.question_lengths:contiguous()
  return data
end

--[[
images: images already preprocessed
labels: question_labels + answer_labels, (D+2)xN

returns a size N Tensor
--]]
function rl_env.produce_reward(images, labels, question_lengths)
  local batch_size = labels:size(2)
  --print(string.format("batch_size:%s", batch_size))
  local image_encodings = cnn:forward(images)
  local conv_feat_maps = cnn:get(30).output:view(batch_size, 512, -1)
  local logprobs = rnn:forward{image_encodings, conv_feat_maps, labels}
  local loss = crit:forward(logprobs, {labels, question_lengths})
  local rewards = -1 * crit.loss_per_sample
  --local w_rescale = 1000
  local w_rescale = 1
  rewards = w_rescale * torch.exp(rewards)
  return rewards
end


return rl_env
