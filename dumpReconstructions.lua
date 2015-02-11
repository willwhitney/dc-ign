require 'cunn'
require 'pl'
require 'paths'
require 'optim'
require 'gnuplot'
require 'math'
require 'rmsprop'
require 'cudnn'
require 'nnx'
require("UnPooling.lua")
require 'mattorch'
require 'image'

bsize = 50
imwidth = 150

TOTALFACES = 5230
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

model = torch.load('logs/vxnet.net')

function load_batch(id, mode)
    collectgarbage()
    local t = mattorch.load('DATASET/' .. mode .. '/' .. id .. '.mat' )
    t=torch.reshape(t.img, torch.LongStorage{bsize, imwidth, imwidth, 3}):double()
    --t = t.img:reshape(bsize, 3, imwidth, imwidth):double()
    return t
end

for t = 1,1 do--num_test_batches do
  -- create mini batch
  local raw_inputs = load_batch(t, 'test')
  mattorch.save('RESULTS/data1.mat', raw_inputs)
  -- local targets = raw_inputs:double()

  -- inputs = raw_inputs:cuda()
  -- -- test samples
  -- local preds = model:forward(inputs)
  -- preds = preds:double()
  
  -- mattorch.save('RESULTS/targets_' .. t .. '.mat', targets )
  -- mattorch.save('RESULTS/preds_' .. t .. '.mat', preds )
  -- print(t, '/', num_test_batches)
end
