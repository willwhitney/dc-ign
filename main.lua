--Tejas Kulkarni (tejask@mit.edu | tejasdkulkarni@gmail.com)
-- Unsupervised Face Synthesis (Conv encoder + decoder)
-- Usage: OMP_NUM_THREADS=1 th main.lua -t 1 -s log 
--so far: th main.lua -t 1 -s log_l20.0001 -p --coefL2 0.0001

--OMP_NUM_THREADS=1 th main.lua -t 1 -r 0.01 --coefL2 0.001 -f 1 -d ../../PASCAL3D/combined/ -p

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
require 'image'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   --dumpTest                                preloads model and dumps .mat for test
   -d,--datasrc       (default "")          data source directory
   -f,--fbmat         (default 0)           load fb.mattorch       
]]

if opt.fbmat == 1 then
  mattorch = require('fb.mattorch')
else
  require 'mattorch'
end

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is a recommended')
end


bsize = 50
imwidth = 150

TOTALFACES = 1000--5230
num_train_batches = 950--5000
num_test_batches =  TOTALFACES-num_train_batches

function load_batch(id, mode)
  return torch.load('DATASET/th_' .. mode .. '/batch' .. id)
end

function init_network()
  local vnet
  
  UMAPS =  24

  vnet = nn.Sequential()
  -------------- ENCODER ---------------
  vnet:add(cudnn.SpatialConvolution(3,UMAPS,11,11,2,2,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialMaxPooling(2,2,1,1))

  vnet:add(cudnn.SpatialConvolution(UMAPS,UMAPS/2,5,5,2,2,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialMaxPooling(2,2,1,1))

  vnet:add(cudnn.SpatialConvolution(UMAPS/2,UMAPS/2,3,3,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialConvolution(UMAPS/2,UMAPS/2,3,3,1,1,1,1))
  vnet:add(cudnn.ReLU())

  vnet:add(cudnn.SpatialConvolution(UMAPS/2,UMAPS/3,3,3,2,2,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialMaxPooling(2,2,1,1))

  vnet:add(nn.View((UMAPS/3)*16*16))
  vnet:add(nn.Dropout(0.5))
  vnet:add(nn.Linear((UMAPS/3)*16*16, 1024))
  vnet:add(cudnn.ReLU())

  -------------- DECODER ---------------
  vnet:add(nn.Dropout(0.5))
  vnet:add(nn.Linear(1024, (UMAPS/3)*16*16))
  vnet:add(cudnn.ReLU())

  vnet:add(nn.View(UMAPS/3,16,16))

  vnet:add(cudnn.SpatialConvolution(UMAPS/3,UMAPS/2,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(nn.SpatialUpSamplingNearest(2))

  vnet:add(cudnn.SpatialConvolution(UMAPS/2,UMAPS/2,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialConvolution(UMAPS/2,UMAPS/2,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())

  vnet:add(cudnn.SpatialConvolution(UMAPS/2,UMAPS,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(nn.SpatialUpSamplingNearest(2))

  vnet:add(cudnn.SpatialConvolution(UMAPS,3,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(nn.SpatialUpSamplingNearest(2))

  --]]
  vnet:cuda()  
  collectgarbage()
  return vnet
end



function test_fw_back(model)
  res=model:forward(training['X'][1]:cuda())
  print(res:size())
  print('Y Size:', training['Y'][1]:size())
  rev=model:backward(training['X'][1]:cuda(),  training['Y'][1]:cuda())
  print(rev:size())
end

model = init_network()
-- test_fw_back(model)
print(model)
parameters,gradParameters = model:getParameters()

criterion = nn.MSECriterion():float()

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
reconstruction = 0



rmsGradAverages = {
  m1W = 1,
  m1b = 1,
  m4W = 1,
  m4b = 1,
  m7W = 1,
  m7b = 1,
  m9W = 1,
  m9b = 1,
  m11W = 1,
  m11b = 1,
  
  m16W = 1,
  m16b = 1,
  
  m19W = 1,
  m19b = 1,
  
  m22W = 1,
  m22b = 1,
  m25W = 1,
  m25b = 1,
  m27W = 1,
  m27b = 1,
  m29W = 1,
  m29b = 1,
  m32W = 1,
  m32b = 1,   
}

--training function
function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- if math.fmod(epoch+1, 50) == 0 then
   --  opt.learningRate = opt.learningRate*0.5
   -- end

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. bsize .. ']')
   for t = 1, num_train_batches do
      -- create mini batch
      local raw_inputs = load_batch(t, 'training')
      local targets = raw_inputs

      inputs = raw_inputs:cuda()

      -- optimize on current mini-batch

      gradParameters:zero()
      -- evaluate function for complete mini batch
      local outputs = model:forward(inputs)
      outputs = outputs:float()
      local f = criterion:forward(outputs, targets)
      
      reconstruction = reconstruction + f

      -- estimate df/dW
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do:cuda())

      -- Stochastic RMSProp on separate layers
      model.modules[1].weight = rmsprop(model.modules[1].weight, model.modules[1].gradWeight,  rmsGradAverages.m1W)
      model.modules[1].bias = rmsprop(model.modules[1].bias, model.modules[1].gradBias,  rmsGradAverages.m1b)

      model.modules[4].weight = rmsprop(model.modules[4].weight, model.modules[4].gradWeight, rmsGradAverages.m4W)
      model.modules[4].bias = rmsprop(model.modules[4].bias, model.modules[4].gradBias,  rmsGradAverages.m4b)

      model.modules[7].weight = rmsprop(model.modules[7].weight, model.modules[7].gradWeight, rmsGradAverages.m7W)
      model.modules[7].bias = rmsprop(model.modules[7].bias, model.modules[7].gradBias,  rmsGradAverages.m7b)

      model.modules[9].weight = rmsprop(model.modules[9].weight, model.modules[9].gradWeight, rmsGradAverages.m9W)
      model.modules[9].bias = rmsprop(model.modules[9].bias, model.modules[9].gradBias,  rmsGradAverages.m9b)

      model.modules[11].weight = rmsprop(model.modules[11].weight, model.modules[11].gradWeight, rmsGradAverages.m11W)
      model.modules[11].bias = rmsprop(model.modules[11].bias, model.modules[11].gradBias,  rmsGradAverages.m11b)
   
      model.modules[16].weight = rmsprop(model.modules[16].weight, model.modules[16].gradWeight, rmsGradAverages.m16W)
      model.modules[16].bias = rmsprop(model.modules[16].bias, model.modules[16].gradBias,  rmsGradAverages.m16b)

      model.modules[19].weight = rmsprop(model.modules[19].weight, model.modules[19].gradWeight, rmsGradAverages.m19W)
      model.modules[19].bias = rmsprop(model.modules[19].bias, model.modules[19].gradBias,  rmsGradAverages.m19b)

      model.modules[22].weight = rmsprop(model.modules[22].weight, model.modules[22].gradWeight, rmsGradAverages.m22W)
      model.modules[22].bias = rmsprop(model.modules[22].bias, model.modules[22].gradBias,  rmsGradAverages.m22b)

      model.modules[25].weight = rmsprop(model.modules[25].weight, model.modules[25].gradWeight, rmsGradAverages.m25W)
      model.modules[25].bias = rmsprop(model.modules[25].bias, model.modules[25].gradBias,  rmsGradAverages.m25b)

      model.modules[27].weight = rmsprop(model.modules[27].weight, model.modules[27].gradWeight, rmsGradAverages.m27W)
      model.modules[27].bias = rmsprop(model.modules[27].bias, model.modules[27].gradBias,  rmsGradAverages.m27b)

      model.modules[29].weight = rmsprop(model.modules[29].weight, model.modules[29].gradWeight, rmsGradAverages.m29W)
      model.modules[29].bias = rmsprop(model.modules[29].bias, model.modules[29].gradBias,  rmsGradAverages.m29b)

      model.modules[32].weight = rmsprop(model.modules[32].weight, model.modules[32].gradWeight, rmsGradAverages.m32W)
      model.modules[32].bias = rmsprop(model.modules[32].bias, model.modules[32].gradBias,  rmsGradAverages.m32b)

      -- disp progress
      xlua.progress(t, num_train_batches)

   end
   
   -- time taken
   time = sys.clock() - time
   time = time / num_train_batches
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   reconstruction = reconstruction / ((num_train_batches))
   print('mean MSE (train set):', reconstruction)
   trainLogger:add{['% mean MSE (train set)'] = reconstruction}
   reconstruction=0

   -- save/log current net
   if math.fmod(epoch, 5) ==0 then
     local filename = paths.concat(opt.save, 'vxnet.net')
     os.execute('mkdir -p ' .. sys.dirname(filename))
     if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
     end
     print('<trainer> saving network to '..filename)
     torch.save(filename, model)
   end
   
   -- next epoch
   epoch = epoch + 1
end



-- test function
function testf()
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')

   reconstruction = 0

   for t = 1,num_test_batches do
      -- create mini batch
      local raw_inputs = load_batch(t, 'test')
      local targets = raw_inputs

      inputs = raw_inputs:cuda()
      -- disp progress
      xlua.progress(t, num_test_batches)

      -- test samples
      local preds = model:forward(inputs)
      preds = preds:float()

      reconstruction = reconstruction + torch.sum(torch.pow(preds-targets,2))
      
      if t == 1 then
        torch.save('tmp/preds', preds)
      end
   end


   -- timing
   time = sys.clock() - time
   time = time / num_test_batches
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   reconstruction = reconstruction / (bsize * num_test_batches * 3 * 150 * 150)
   print('mean MSE error (test set)', reconstruction)
   testLogger:add{['% mean class accuracy (test set)'] = reconstruction}
   reconstruction = 0
end


----------------------------------------------------------------------
-- and train!
--

tcounter = 0
while true do
   -- train/test
  train()
  if math.fmod(tcounter,5) == 0 then
    testf()
  end

 -- plot errors
 if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()
 end
end
--]]
