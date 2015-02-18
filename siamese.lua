
-- works well: th finetune.lua -t 1 -s log_FINE_TUNE2 -r 0.1 --coefL2 0.001 -m 0.9
-- take a trained model and output features for all the behavioral test images
require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'rmsprop'
require 'KLDCriterion'
require 'LinearCR'
require 'Reparametrize'
require 'cutorch'
require 'cunn'
require 'optim' 
require 'GaussianCriterion'
require 'testf'
require 'utils'
require 'config'
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
   -c,--color         (default 0)           color or not 
   -u,--reuse	      (default 0)           reuse existing network weights
]]

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

opt.cuda = true


MODE_TEST = 'FT_test'
MODE_TRAINING = 'FT_training'
model = torch.load('log150gl001F96_OPENMIND/vxnet.net')
ENC_DIM = model:get(1).modules[10].output:size()[2]
print(ENC_DIM)
print(model)

-- get training and test data
if false then
	bsize = 0
   ALLDATA = {}
   cnt=1
   num_faces = 200
   for i=1,num_faces do
      collectgarbage()
      local batch = load_batch(i, MODE_TRAINING)
      model:forward(batch:cuda())
      ALLDATA[i] = {}
      ftrs = model:get(1).modules[10].output:double() 
      for j=1,batch:size()[1] do
         ALLDATA[i][j] =  ftrs[j]
      end
      bsize = batch:size()[1]
      print(i,'/', num_faces)
   end

   num_training = num_faces*((bsize-1) + 2*bsize)

	training = {}
	training['X'] = torch.zeros(num_training, 2*ENC_DIM)
	training['Y'] = torch.zeros(num_training) 
	tr_id = 1
	for i=1, num_faces do
		for cnt=2,bsize do
			training['X'][{{tr_id},{1,ENC_DIM}}] = ALLDATA[i][1]
         training['X'][{{tr_id},{ENC_DIM+1,2*ENC_DIM}}] = ALLDATA[i][cnt]
			training['Y'][tr_id] = 2
			tr_id = tr_id + 1
		end
      for j=1,2*bsize do
         training['X'][{{tr_id},{1,ENC_DIM}}] = ALLDATA[i][1]
         training['X'][{{tr_id},{ENC_DIM+1,2*ENC_DIM}}] = ALLDATA[torch.random(num_faces)][torch.random(bsize)]
         training['Y'][tr_id] = 1
         tr_id = tr_id + 1
      end
		print('face:', i,'/',num_training, ' tr_id:', tr_id)
	end

	torch.save('DATASET/FINETUNE/training.t7', training)

   print('Generating test set from behavioral data ...')
   test = {}
   num_test = 96
   test['X'] = torch.zeros(num_test, 2*ENC_DIM)
   test['Y'] = torch.Tensor({   1,  1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0,  0,  1,  1,  1,
                                1,  1,  1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1,  0,  1,  1,  0,
                                0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0,
                                1,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  1,
                                1,  0,  1,  0,  0,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  1,  0,  1,
                                0,  1,  1,  0,  0,  0})

   test['Y'] = test['Y'] + 1

   for id=1,96 do
      local ftrs1 = 0;
      local ftrs2 = 0;
      for pid = 0,1 do
         fname = 'DATASET/images_wbg/' .. 'pair_' .. id .. '_' .. pid .. '.png'
         im_tmp = image.load(fname)
         im = torch.zeros(1,256, 256)
         im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
         im = image.scale(im[1], imwidth ,imwidth)
         im = torch.reshape(im, 1, imwidth, imwidth)
         batch = torch.zeros(1,1,imwidth,imwidth)  
         batch[1] = im
         res = model:forward(batch:cuda())
         ftrs= model:get(1).modules[10].output:double() 
         if pid == 0 then
            ftrs1 = ftrs
         else
            ftrs2 = ftrs
         end
      end
      test['X'][{{id},{1,ENC_DIM}}] = ftrs1
      test['X'][{{id},{ENC_DIM+1,2*ENC_DIM}}] = ftrs2
   end
	torch.save('DATASET/FINETUNE/test.t7', test)
else
	print('Loading cached data ...')
	training = torch.load('DATASET/FINETUNE/training.t7')
	test = torch.load('DATASET/FINETUNE/test.t7')	
end


num_training = (#training['Y'])[1]
num_test = (#test['Y'])[1]

classes = {1,2}

classifier = nn.Sequential()
-- classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(ENC_DIM*2, 1000))
classifier:add(nn.ReLU())

classifier:add(nn.Linear(1000, 1000))
classifier:add(nn.ReLU())

-- -- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(2000, 500))
-- classifier:add(nn.ReLU())

-- -- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(500, 50))
-- classifier:add(nn.ReLU())

classifier:add(nn.Linear(1000, #classes))
classifier:add(nn.LogSoftMax())
classifier:cuda()


-- res = classifier:forward(load_batch(1, MODE_TEST):cuda())
-- print(res:size())


-- retrieve parameters and gradients
parameters,gradParameters = classifier:getParameters()

criterion = nn.ClassNLLCriterion()

criterion:cuda()


-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

print('num_train_batches:', num_train_batches)
batchSize = 100
print(num_training)
print('Started training ... ')
print('Learning rate:', opt.learningRate)
num_train_batches = (num_training/batchSize)
-- num_test_batches = (num_test/batchSize)

-- training function
function trainf()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- if epoch > 10 then
   -- 	opt.learningRate = opt.learningRate*0.1
   -- end
   print("LR:", opt.learningRate)
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   local indxs = torch.randperm(num_training)
   tr_err = 0
   for t = 1,num_train_batches do
      -- create mini batch
      -- local inputs = load_batch(t, MODE_TRAINING)
      -- local targets = torch.ones(batchSize,1)*t --same identities in every batch. change this later
	  
	  local inputs = torch.zeros(batchSize, ENC_DIM*2)
	  local targets = torch.zeros(batchSize,1)
	  for bid=1,batchSize do
	  	jj = indxs[(t-1)*batchSize +bid]
	  	inputs[bid] = training['X'][jj]
	  	targets[bid] = training['Y'][jj]
	  end      

      inputs = inputs:cuda()
      targets = targets:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = classifier:forward(inputs)

         local f = criterion:forward(outputs, targets)
         -- print('OUT', outputs:size())
         -- print(outputs[1])
         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         classifier:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- -- update confusion
	      for i = 1,batchSize do
	        -- confusion:add(preds[i]:float(), targets[i][1])
	        predexp = torch.exp(outputs[i]:float())
	        maxv,ind = torch.max(predexp,1)
         --   if t==1 then
	        --    print('target:', targets[i][1], ' pred:', ind[1])
	        -- end
           if targets[i][1] ~= ind[1] then
	        	tr_err = tr_err + 1
	        end
	      end
         -- print("\nGRAD:", gradParameters:sum())
         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            -- learningRateDecay = 5e-4--5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, num_train_batches)

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / num_train_batches
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   -- print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = tr_err}--confusion.totalValid * 100}
   -- confusion:zero()
   print('<TRAINING> Error:' ,tr_err)

   -- save/log current net
   if true then--math.fmod(epoch, 5) == 0 then
	   local filename = paths.concat(opt.save, 'classifier.net')
	   os.execute('mkdir -p ' .. sys.dirname(filename))
	   if paths.filep(filename) then
	      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
	   end
	   print('<trainer> saving network to '..filename)
	   torch.save(filename, classifier)
   end

   -- next epoch
   epoch = epoch + 1
end

-- test function
function testf()
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   tst_err = 0
   print('<trainer> on testing Set:')
   for t = 1,1 do
      -- disp progress
      xlua.progress(t, num_test_batches)

	  local inputs = test['X']--torch.zeros(batchSize, ENC_DIM*2)
	  local targets = test['Y']--torch.zeros(batchSize, 1)
	  -- for bid=1,batchSize do
	  -- 	jj = (t-1)*batchSize +bid
	  -- 	inputs[bid] = test['X'][jj]
	  -- 	targets[bid] = test['Y'][jj]
	  -- end    
	  inputs = inputs:cuda()
	  targets = targets:cuda()
      -- -- create mini batch
      -- local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      -- local targets = torch.Tensor(opt.batchSize)

      -- test samples
      local preds = classifier:forward(inputs)
      for i = 1,preds:size()[1] do
        -- confusion:add(preds[i]:float(), targets[i][1])
        predexp = torch.exp(preds[i]:float())
        maxv,ind = torch.max(predexp,1)
        -- print('target:', targets[i], ' pred:', ind[1])
        if targets[i] ~= ind[1] then
        	tst_err = tst_err + 1
        end
      end
      -- print('--------------')
      
   end

   -- timing
   time = sys.clock() - time
   time = time / num_test_batches
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   -- print(confusion)
   print('<TEST> Errors:', tst_err, ' accuracy:', (1-(tst_err/num_test))*100)--confusion.totalValid*100)
   testLogger:add{['% mean class error (test set)'] = tst_err}--confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   trainf()
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