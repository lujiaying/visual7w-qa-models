require 'nn'

-------------------------------------------------------------------------------
-- Policy Gradient QA Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.QAPGCriterion', 'nn.Criterion')
function crit:__init(env)
  parent.__init(self)

  -- Environment to produce rewards
  self.env = env
end

--[[
input: {images(NxCxHxW), labels(DxN), question_lengths(N)}
targets: nil

returns a float number
--]]
function crit:updateOutput(input, targets)
  local images, labels, question_lengths = unpack(input)
  local batch_size = labels:size(2)
  local image_encodings = self.env.cnn:forward(images)
  local conv_feat_maps = self.env.cnn:get(30).output:clone()
  conv_feat_maps = conv_feat_maps:view(batch_size, 512, -1)
  local logprobs = self.env.rnn:forward{image_encodings, conv_feat_maps, labels}
  self.output = self.env.crit:forward(logprobs, {labels, question_lengths})
  self.reward_per_sample = env.crit.loss_per_sample
  -- negative loss is reward, size N
  self.reward_per_sample = self.reward_per_sample:cmul(torch.Tensor(batch_size):fill(-1))
  return self.output
end

--[[
input: {logprobs((LxNx(M+1)), labels(DxN), question_lengths(N)}, L=D+2
gradOutput: nil

returns a (D+2)xNx(M+1) Tensor.
--]]
--function crit:updateGradInput(logprobs, labels, question_lengths)
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
        local one_hot = logprobs[{ t,b }]:clone():zero()
        one_hot[target_index] = 1
        print(string.format('logprobs[{t,b}] size:%s, one_hot size:%s', logprobs[{t,b}]:size(), one_hot:size()))
        print(string.format('reward: %s', reward))
        self.gradInput[{ t,b }] = reward * (logprobs[{ t,b }] - one_hot)
      end
    end
  end
  return self.gradInput
end
