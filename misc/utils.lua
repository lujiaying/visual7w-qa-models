local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read("*all")
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

-- split the labels into two vectors of question and answer 
function utils.split_question_answer(labels, question_length, answer_length, question_seq_length, answer_seq_length)
  local batch_size = labels:size()[2]
  local q_length = question_seq_length or question_length:max()
  local a_length = answer_seq_length or answer_length:max()
  local question_labels = torch.LongTensor(q_length, batch_size)
  local answer_labels = torch.LongTensor(a_length, batch_size)
  question_labels:zero()
  answer_labels:zero()
  for i = 1, batch_size do
    local lq = question_length[i]
    local la = answer_length[i]
    question_labels[{{1,lq}, {i,i}}] = labels[{{1,lq}, {i,i}}]
    answer_labels[{{1,la}, {i,i}}] = labels[{{1+lq,lq+la}, {i,i}}]
  end
  return question_labels, answer_labels
end

function utils.pack_question_answer(question_labels, q_lens, answer_labels, a_lens, seq_len)
  assert(question_labels:size()[2] == answer_labels:size()[2], 'error: question answer batch size NOT equal')
  local batch_size = question_labels:size()[2]
  local max_q_len = q_lens:max()
  local max_a_len = a_lens:max()
  if max_q_len + max_a_len > seq_len then
    print(string.format('q_len:%s, a_len:%s, seq_len:%s', max_q_len, max_a_len, seq_len))
  end
  local labels = torch.LongTensor(batch_size, seq_len):zero()
  for k=1, batch_size do
    local seq = torch.LongTensor(seq_len):zero()
    local question_label = question_labels[{ {}, k }]
    seq[{{1, q_lens[k]}}] = question_label[{{1, q_lens[k]}}]
    local answer_label = answer_labels[{ {}, k }]
    seq[{{q_lens[k]+1, q_lens[k]+a_lens[k]}}] = answer_label[{{1, a_lens[k]}}]
    labels[k] = seq
  end
  labels = labels:transpose(1,2):contiguous()
  return labels
end

--[[
answer_logprobs: size (D+2)xN, which only contains answer tokens
answer_lengths: size N
gpu_mode: boolean

returns a N Tensor.
--]]
function utils.cal_answer_sum_logp(answer_logprobs, answer_lengths, gpu_mode)
  if gpu_mode == nil then gpu_mode = false end
  local L = answer_logprobs:size(1)
  local D = L-2
  local N = answer_logprobs:size(2)
  local seq_sum_logprobs = torch.FloatTensor(N):zero()
  if gpu_mode then seq_sum_logprobs:cuda() end

  for b=1, N do
    for t=1, answer_lengths[b] do
      seq_sum_logprobs[b] = seq_sum_logprobs[b] + answer_logprobs[{t,b}]
    end
  end
  return seq_sum_logprobs
end

--[[
sequences: a list of string

returns a size #sequences tensor
]]--
function utils.get_sequences_token_cnt(sequences)
  local token_cnts = torch.zeros(#sequences)
  for k = 1, #sequences do
    local token_cnt = 0
    local sequence = sequences[k]
    sequence = string.lower(sequence):gsub('%p', '')
    for token in sequence:gmatch('%w+') do
      token_cnt = token_cnt + 1
    end
    token_cnts[k] = token_cnt
  end
  return token_cnts
end

return utils
