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

TOTALFACES = 5231
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

-- if PRELOAD == 1 then
--   training = torch.load('face_dataset_training')
--   test = torch.load('face_dataset_test')
--   num_train_batches = training:size()[1]
--   num_test_batches = test:size()[1]
-- else
--   --load all batches
--   -- training = {}; training['X'] = {}; training['Y'] = {}
--   -- validation = {}; validation['X'] = {}; validation['Y'] = {}
--   -- test = {}; test['X'] = {}; test['Y'] = {};

--   basedir = '../facemachine/CNN_DATASET/'

--   TOTALFACES = 5231
--   num_train_batches = 5000
--   num_test_batches =  TOTALFACES-num_train_batches
  
--   training = torch.Tensor(num_train_batches, bsize, 3, imwidth, imwidth)
--   test = torch.Tensor(num_test_batches, bsize, 3, imwidth, imwidth)

--   for i=1,num_train_batches do
--     print(i, '/', num_train_batches)
--     local t = mattorch.load(basedir .. 'face_' .. i .. '/data' .. '.mat' )
--     t = t.img:reshape(bsize, 3, imwidth, imwidth):float()/255
--     t = torch.Tensor(t:size()):copy(t)
--     training[i] = t
--   end

--   for i=1,num_test_batches do
--     local FID = i + num_train_batches
--     local t = mattorch.load(basedir .. 'face_' .. FID .. '/data' .. '.mat' )
--     t = t.img:reshape(bsize, 3, imwidth, imwidth):float()/255
--     t = torch.Tensor(t:size()):copy(t)
--     test[i] = t
--   end
-- end

function load_batch(id)
    basedir = '../facemachine/CNN_DATASET/'
    local t = mattorch.load(basedir .. 'face_' .. id .. '/data' .. '.mat' )
    t = t.img:reshape(bsize, 3, imwidth, imwidth):float()/255
    t = torch.Tensor(t:size()):copy(t)
    return t
end

function init_network()
  local vnet
  
  vnet = nn.Sequential()
  -------------- ENCODER ---------------
  vnet:add(cudnn.SpatialConvolution(3,32,11,11,2,2,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialMaxPooling(2,2,1,1))

  vnet:add(cudnn.SpatialConvolution(32,16,5,5,2,2,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialMaxPooling(2,2,1,1))

  vnet:add(cudnn.SpatialConvolution(16,16,3,3,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialConvolution(16,16,3,3,1,1,1,1))
  vnet:add(cudnn.ReLU())

  vnet:add(cudnn.SpatialConvolution(16,8,3,3,2,2,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialMaxPooling(2,2,1,1))

  vnet:add(nn.View(8*16*16))
  vnet:add(nn.Linear(8*16*16, 1024))
  vnet:add(cudnn.ReLU())

  -------------- DECODER ---------------
  vnet:add(nn.Linear(1024, 8*16*16))
  vnet:add(cudnn.ReLU())

  vnet:add(nn.View(8,16,16))

  vnet:add(cudnn.SpatialConvolution(8,16,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(nn.SpatialUpSamplingNearest(2))

  vnet:add(cudnn.SpatialConvolution(16,16,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(cudnn.SpatialConvolution(16,16,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())

  vnet:add(cudnn.SpatialConvolution(16,32,2,2,1,1,1,1))
  vnet:add(cudnn.ReLU())
  vnet:add(nn.SpatialUpSamplingNearest(2))

  vnet:add(cudnn.SpatialConvolution(32,3,2,2,1,1,1,1))
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
  
  m15W = 1,
  m15b = 1,
  
  m17W = 1,
  m17b = 1,
  
  m20W = 1,
  m20b = 1,
  m23W = 1,
  m23b = 1,
  m25W = 1,
  m25b = 1,
  m27W = 1,
  m27b = 1,
  m30W = 1,
  m30b = 1,   
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
      local raw_inputs = load_batch(t)
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

      model.modules[15].weight = rmsprop(model.modules[15].weight, model.modules[15].gradWeight, rmsGradAverages.m15W)
      model.modules[15].bias = rmsprop(model.modules[15].bias, model.modules[15].gradBias,  rmsGradAverages.m15b)

      model.modules[17].weight = rmsprop(model.modules[17].weight, model.modules[17].gradWeight, rmsGradAverages.m17W)
      model.modules[17].bias = rmsprop(model.modules[17].bias, model.modules[17].gradBias,  rmsGradAverages.m17b)

      model.modules[20].weight = rmsprop(model.modules[20].weight, model.modules[20].gradWeight, rmsGradAverages.m20W)
      model.modules[20].bias = rmsprop(model.modules[20].bias, model.modules[20].gradBias,  rmsGradAverages.m20b)

      model.modules[23].weight = rmsprop(model.modules[23].weight, model.modules[23].gradWeight, rmsGradAverages.m23W)
      model.modules[23].bias = rmsprop(model.modules[23].bias, model.modules[23].gradBias,  rmsGradAverages.m23b)

      model.modules[25].weight = rmsprop(model.modules[25].weight, model.modules[25].gradWeight, rmsGradAverages.m25W)
      model.modules[25].bias = rmsprop(model.modules[25].bias, model.modules[25].gradBias,  rmsGradAverages.m25b)

      model.modules[27].weight = rmsprop(model.modules[27].weight, model.modules[27].gradWeight, rmsGradAverages.m27W)
      model.modules[27].bias = rmsprop(model.modules[27].bias, model.modules[27].gradBias,  rmsGradAverages.m27b)

      model.modules[30].weight = rmsprop(model.modules[30].weight, model.modules[30].gradWeight, rmsGradAverages.m30W)
      model.modules[30].bias = rmsprop(model.modules[30].bias, model.modules[30].gradBias,  rmsGradAverages.m30b)

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
   if math.fmod(epoch, 10) ==0 then
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
      local raw_inputs = load_batch(t + num_train_batches)
      local targets = raw_inputs

      inputs = raw_inputs:cuda()
      -- disp progress
      xlua.progress(t, num_test_batches)

      -- test samples
      local preds = model:forward(inputs)
      preds = preds:float()

      reconstruction = reconstruction + torch.sum(torch.pow(preds-targets,2))
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

-- torch.save('face_dataset_training', training)
-- torch.save('face_dataset_test', test)


while true do
   -- train/test
   train()
   testf()

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end
--]]