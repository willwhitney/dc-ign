-- Torch interface to Julia (server)
local json = require ("dkjson")
local zmq = require "lzmq"
local context = zmq.init(1)

cmd = torch.CmdLine()
cmd:option('-port', 5000, 'port number')
params = cmd:parse()
PORT = params.port

local socket = context:socket(zmq.REP)
socket:bind("tcp://*:" .. PORT)

function serialize(data)
	if torch.type(1) == torch.type(data) then
		return {data}
	end
	--for now it assumes data is a 1-D array
	local ret = {}
	for i=1,data:size(1) do
	  ret[i] = data[i]
	end
	return ret
end

-- function test()
-- 	require("picture.lua")
-- 	getfeatures()
-- end



while true do
    --  Wait for next request from client
    local request = socket:recv()
    -- print("Received Hello [" .. request .. "]")
    -- print(request)
    if request ~= nil then
	    request = json.decode(request, 1, nil)
	    -- print('req:',request, ' | cmd:', request.cmd)
	    if request.cmd == "load" then
	    	require(request.name)
	    	ret = {1}
		    ret = json.encode (ret, { indent = true })
		    socket:send(ret)
		elseif request.cmd == "call" then
			func_name = request.msg.func
			args = request.msg.args
			ret = _G[func_name](args)[1]
			-- ret = torch.rand(10):float()
		    ret = serialize(ret)
		    -- print(ret)
		    ret = json.encode (ret,{ indent = true })
		    socket:send(ret)
		end
	end
end
--  We never get here but if we did, this would be how we end
socket:close()
context:term()


--]]
