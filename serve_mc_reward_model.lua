local http = require('http')
require 'torch'
local cjson = require 'cjson'
local rl_env = require 'rl_env'

local gpuid = 3
rl_env.init(gpuid)

local ip_address = '10.218.105.43'
local port = 1337

http.createServer(function (req, res)
  local req_body = ""
  req:on('data', function(chunk) 
    req_body = req_body.. chunk
  end)

  req:on("end", function()
    local req_table = cjson.decode(req_body) -- image_ids, qa_ids, answers

    local data = rl_env.get_features(req_table.image_ids, req_table.questions, req_table.answers)
    local rewards = rl_env.produce_reward(data.images, data.labels, data.question_lengths)

    local body = cjson.encode(rewards:totable())
    --print(body)
    res:setHeader("Content-Type", "text/plain")
    res:setHeader("Content-Length", #body)
    res:finish(body)
  end)

end):listen(port, ip_address)

print(string.format('Server running at http://%s:%s/', ip_address, port))


--[[
# python request
r = requests.post('http://10.218.105.43:1337/', data=json.dumps({'image_ids':[1, 1, 1], 'qa_ids':[127335, 102156, 102155]
...: , 'questions':['How are the cars parked?', 'What is the street lined with?', 'What are the people doing?'], 'answers':['I
...: n a line.', 'Parking meters.', 'Talking.']}))
--]]
