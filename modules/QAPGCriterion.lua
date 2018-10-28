require 'nn'
require 'math'

-------------------------------------------------------------------------------
-- Policy Gradient QA Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.QAPGCriterion', 'nn.Criterion')
function crit:__init(env, gpu_mode)
  parent.__init(self)

  -- Environment to produce rewards
  self.env = env
  self.gpu_mode = gpu_mode -- true or false
end

--[[
input: {images(NxCxHxW), labels(DxN), question_lengths(N), sample_sum_logprobs(N)}
targets: nil

returns a float number
--]]
function crit:updateOutput(input, targets)
  local images, labels, question_lengths, sample_sum_logprobs = unpack(input)
  local batch_size = labels:size(2)
  local image_encodings = self.env.cnn:forward(images)
  local conv_feat_maps = self.env.cnn:get(30).output:clone()
  conv_feat_maps = conv_feat_maps:view(batch_size, 512, -1)
  local qa_logprobs = self.env.rnn:forward{image_encodings, conv_feat_maps, labels}
  self.env.crit:forward(qa_logprobs, {labels, question_lengths})

  self.reward_per_sample = -1 * env.crit.loss_per_sample
  print(string.format('before rescale reward: %s \n\n', self.reward_per_sample))
  -- reward = -1 * MLE_loss, size N, equals to logprob of sequence;
  -- then scale reward to a positive num; w_rescale * exp(reward)
  local w_rescale = 10
  self.reward_per_sample = w_rescale * torch.exp(self.reward_per_sample)

  -- loss/output = sum(-reward * sample_sum_logp) / batch_size
  print(string.format('reward and logprobs: %s \n %s\n\n', self.reward_per_sample, sample_sum_logprobs))
  self.output = torch.sum( (-1*self.reward_per_sample:clone()):cmul(sample_sum_logprobs) )
  self.output = self.output / batch_size
  return self.output
end

--[[
input: {logprobs((LxNx(M+1)), labels(DxN), question_lengths(N)}, L=D+2
gradOutput: nil

returns a (D+2)xNx(M+1) Tensor.
--]]
function crit:updateGradInput(input, gradOutput)
  local logprobs, labels, question_lengths = unpack(input)
  local L, N, Mp1 = logprobs:size(1), logprobs:size(2), logprobs:size(3)
  local D = labels:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  self.gradInput:resizeAs(logprobs):zero()

  for b = 1, N do  -- iterate over batches
    local first_time = true
    local reward = self.reward_per_sample[b]
    for t = question_lengths[b] + 2, L do -- iterate over sequence time
      local target_index
      if t-1 > D then
        target_index = 0
      else
        target_index = labels[{ t-1,b }] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end
      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        print(string.format('gradInput[{%s,%s,%s}] reward:%s', t, b, target_index, reward))
        self.gradInput[{ t,b,target_index }] = - reward
      end
    end
  end
  return self.gradInput
end
