require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'
require 'loadcaffe'

-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'misc.QADatasetLoader'
require 'modules.QAAttentionModel'
require 'modules.QACriterion'


-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
torch.manualSeed(1234)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
--gpu_mode = opt.gpuid >= 0
gpu_mode = true

if gpu_mode then
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  --if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(1234)
  cutorch.setDevice(1)
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
local checkpoint = torch.load("./model_visual7w_telling_gpu.t7")
local modules = checkpoint.modules
-- modules.crit = nn.QACriterion()
-- modules.rnn:createClones() -- reconstruct clones inside the language model
if gpu_mode then for k,v in pairs(modules) do v:cuda() end end

collectgarbage() -- free some memory

-------------------------------------------------------------------------------
-- Get word embeddings
-------------------------------------------------------------------------------
local module_list = modules.rnn:getModulesList()
local lookup_table = module_list[2]
--print(module_list)

local emb_matrix = lookup_table:parameters()[1]
print(emb_matrix:size())
file = io.open("./data/telling_gpu_word_emb_parameters.txt", 'w')
for i=1, emb_matrix:size()[1] do
    file:write(table.concat(torch.totable(emb_matrix[i]), " "))
    file:write("\n")
end
file:close()
