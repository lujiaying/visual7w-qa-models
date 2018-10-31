require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'image'

require 'modules.QAAttentionModel'
require 'modules.QACriterion'
require 'modules.QAPGCriterion'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Demo image and questions
-------------------------------------------------------------------------------

-- You can replace it with your own image
--local image_file = 'data/demo.jpg'
local image_file = 'images/v7w_1.jpg'

-- You can write your own questions here
--local questions = {
--  'how many people are there?',
--  'what animal can be seen in the picture?',
--  'who is wearing a red shirt?',
--  'what color is the elephant?',
--  'when is the picture taken?'
--}
local questions = {
  "What color is the sidewalk?",
  "Where are the men talking?",
  "What is on the sidewalk's edge?"
}
--[[
local questions = {
  "What color is the sidewalk?",
  "Where are the men talking?",
  "What is on the sidewalk's edge?",
  "How many cars are parked?",
  "What is on the sidewalk?",
  "How is the sidewalk paved?",
  "What is on the sidewalk?",
  "Where are trees?",
  "Who is wearing black pants?",
  "Who is wearing a gray jacket?",
  "What are the people doing?",
  "What is the street lined with?",
  "How are the cars parked?"
}
]]--

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
modules = nil
collectgarbage() -- free some memory

if gpu_mode then cnn:cuda() end
rnn:createClones()
if gpu_mode then rnn:cuda() end

-- prepare QA policy gradient criterion
local env = {}
env['rnn'] = rnn:clone()
env['cnn'] = cnn:clone()
env['crit'] = nn.QACriterion()
if gpu_mode then
  for k,v in pairs(env) do v:cuda() end
end
env.cnn:evaluate()
env.rnn:evaluate()
local qa_pg_crit = nn.QAPGCriterion(env)
if gpu_mode then
  qa_pg_crit:cuda()
end

local function get_sequence_token_cnt(sequence)
  local token_cnt = 0
  for token in sequence:gmatch('%w+') do
    token_cnt = token_cnt + 1
  end
  return token_cnt
end

-------------------------------------------------------------------------------
-- Prepare inputs
-------------------------------------------------------------------------------
print(string.format('vocab size:%s', vocab_size))
local MAX_Q_LEN = 15
local num_q = #questions
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
print(string.format('Question labels: %s', question_labels))
print(string.format('Question lens: %s', q_len))

-------------------------------------------------------------------------------
-- Start demo
-------------------------------------------------------------------------------
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

cnn = nil
collectgarbage() -- free some memory

-- forward the model to also get generated samples for each image
local answer_labels, seq_logprobs, seq_logprobs_pertime = rnn:sample(image_encodings, conv_feat_maps, question_labels, q_len)
local answers = net_utils.decode_sequence(vocab, answer_labels)
print(string.format("seq_logprobs size:%s, seq_logprobs_pertime size:%s", seq_logprobs:size(), seq_logprobs_pertime:size()))
print(string.format("answer_labels: %s", answer_labels))

local a_len = torch.LongTensor(num_q):zero()
for k = 1, #answers do
  local cur_a_len = get_sequence_token_cnt(answers[k])
  a_len[k] = cur_a_len
end
local answer_sum_logprobs = utils.cal_answer_sum_logp(seq_logprobs, a_len, gpu_mode)

rnn = nil
collectgarbage() -- free some memory

-- calculate policy gradient loss
print(string.format('img size:%s', img:size()))
local images = torch.Tensor(num_q, img:size(2), img:size(3), img:size(4)):cuda()
print(string.format('images size:%s', images:size()))
local max_q_len = 15 --15 in train
local max_a_len = 5
local labels = torch.LongTensor(#answers, max_q_len+max_a_len):zero()  -- question + answer
for k=1, num_q do
  local seq = torch.LongTensor(max_q_len+max_a_len):zero()
  local question_label = question_labels[{ {}, k }]
  seq[{{1, q_len[k]}}] = question_label[{{1, q_len[k]}}]
  -- seq[q_len[k]+1] = vocab_size   -- Already add <START> token to original question
  local answer_label = answer_labels[{ {}, k }]
  seq[{{q_len[k]+1, q_len[k]+a_len[k]}}] = answer_label[{{1, a_len[k]}}]
  labels[k] = seq
  images[k] = img:clone()
end
print(string.format('labels(q+a): %s', labels))
labels = labels:transpose(1,2):contiguous()
local pg_loss = qa_pg_crit:forward({images, labels, q_len:contiguous(), answer_sum_logprobs}, {})
print(string.format('images size:%s', images:size()))
print(string.format('answer lens:%s', a_len))
print(string.format('policy_gradient loss:%s', pg_loss))
if gpu_mode then seq_logprobs_pertime = seq_logprobs_pertime:cuda() end
local dlogprobs = qa_pg_crit:backward({seq_logprobs_pertime, labels, q_len:contiguous()}, {})
print(string.format('dlogprobs size:%s', dlogprobs:size()))

-------------------------------------------------------------------------------
-- Output results
-------------------------------------------------------------------------------
local questions = net_utils.decode_sequence(vocab, question_labels)
print(string.format("answer_labels size:%s", answer_labels:size()))
print('** QA demo on ' .. image_file .. ' **\n')
for k = 1, #answers do
  print(string.format('Q: %s ?', questions[k]))
  print(string.format('A: %s .', answers[k]))
  local actual_seq_logprobs = {}
  for t = 1, a_len[k] do
    table.insert(actual_seq_logprobs, seq_logprobs[{t, k}])
  end
  print( string.format("Answer len:%s, logprobs: %s", answer_len, table.concat(actual_seq_logprobs, ", ")) )
  print('\n')
end
