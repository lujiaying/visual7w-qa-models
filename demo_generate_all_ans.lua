require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'image'

require 'modules.QAAttentionModel'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image QA demo')
cmd:text()
cmd:text('Options')

cmd:option('-model', 'checkpoints/model_visual7w_telling_cpu.t7', 'path to model to evaluate')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1234, 'random number generator seed to use')

local opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Basic setup and load pretrianed model
-------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
gpu_mode = opt.gpuid >= 0

if gpu_mode then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

assert(string.len(opt.model) > 0, 'error: must provide a model')
local checkpoint = torch.load(opt.model)
local vocab = checkpoint.vocab
local vocab_size = 0
local word_to_ix = {}
for i, w in pairs(vocab) do
  word_to_ix[w] = i
  vocab_size = vocab_size + 1
end

local modules = checkpoint.modules
local cnn = modules.cnn
local rnn = modules.rnn
rnn:createClones()
if gpu_mode then cnn:cuda() end
if gpu_mode then rnn:cuda() end
modules = nil
collectgarbage() -- free some memory

-------------------------------------------------------------------------------
-- Produce answer
-------------------------------------------------------------------------------

local function produce_answer(image_file, questions, cnn, rnn, sample_num)
  -- Preapare inputs --
  local MAX_Q_LEN = 30
  local num_q = #questions
  for k, q in pairs(questions) do
    local s = string.lower(q):gsub('%p', '')
    local q_actual_len = 0
    for token in s:gmatch('%w+') do
      q_actual_len = q_actual_len + 1
    end
    if q_actual_len > MAX_Q_LEN then MAX_Q_LEN = q_actual_len + 5 end
  end
  local question_labels = torch.LongTensor(MAX_Q_LEN, num_q):zero()
  local q_len = torch.LongTensor(num_q):zero()
  for k, q in pairs(questions) do
    local s = string.lower(q):gsub('%p', '')
    for token in s:gmatch('%w+') do
      if q_len[k] < MAX_Q_LEN and word_to_ix[token] then
        q_len[k] = q_len[k] + 1
        question_labels[q_len[k]][k] = word_to_ix[token]
      end
    end
    q_len[k] = q_len[k] + 1
    question_labels[q_len[k]][k] = vocab_size
  end

  -- Start demo --
  -- forward CNN
  local img = image.load(image_file, 3)
  img = image.scale(img, 224, 224, 'bicubic'):view(1, 3, 224, 224):mul(255)
  img = net_utils.prepro(img, false, gpu_mode)
  
  -- encode image and question tokens with a pretrained LSTM
  local image_encodings = cnn:forward(img)
  
  -- convolutional feature maps for attention
  -- layer #30 in VGG outputs the 14x14 conv5 features
  local conv_feat_maps = cnn:get(30).output:clone()
  conv_feat_maps = conv_feat_maps:view(1, 512, -1)
  
  image_encodings = torch.repeatTensor(image_encodings, num_q, 1)
  conv_feat_maps = torch.repeatTensor(conv_feat_maps, num_q, 1, 1)
  
  -- forward the model to also get generated samples for each image
  local opt_sample = {}
  opt_sample['sample_max'] = sample_num
  local answer_tokens_list = {}
  for t=1, sample_num do
    local answer_labels = rnn:sample(image_encodings, conv_feat_maps, question_labels, q_len, opt_sample)
    local answers = net_utils.decode_sequence(vocab, answer_labels)
    table.insert(answer_tokens_list, answers)
  end
  return answer_tokens_list
end

-------------------------------------------------------------------------------
-- Main process
-------------------------------------------------------------------------------
local json_path = './visual7w-toolkit/datasets/visual7w-telling/dataset.json'
local info = utils.read_json(json_path)
local result_file = io.open('./data/telling_generation.txt', 'w')

for idx, image in ipairs(info.images) do
  local image_id = image.image_id
  if image.split == 'test' then
    local questions = {}
    local qa_ids = {}
    local golden_ans = {}
    for qa_idx, qa_pair in ipairs(image.qa_pairs) do
      table.insert(questions, qa_pair.question)
      table.insert(qa_ids, qa_pair.qa_id)
      table.insert(golden_ans, qa_pair.answer)
    end
    local image_path = string.format('./images/v7w_%s.jpg', image_id)
    -- print(idx, image_path)
    local sample_num = 5
    local answers_list = produce_answer(image_path, questions, cnn, rnn, sample_num)
    for k = 1, #questions do
      local generated_ans = {}
      for s_idx = 1, sample_num do
          table.insert(generated_ans, answers_list[s_idx][k])
      end
      result_file:write(string.format("%s\t%s\t%s\t%s\t%s\n", image_id, qa_ids[k], questions[k], golden_ans[k], table.concat(generated_ans, '\001')) )
    end
  end
end
result_file:close()
