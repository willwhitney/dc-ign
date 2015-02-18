--supervised fine-tuning
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
model = torch.load('log_NEW_init_network2_150_F40_H60/vxnet.net')
encoder = model:get(1)
print(encoder)
num_faces= 150--200
-- get training and test data
if true then
	
	training = {}
	training['X'] = torch.zeros(num_faces*25,1,150,150):float() -- * 40
	training['Y'] = torch.zeros(num_faces*25) -- * 40
	-- test = {}
	-- test['X'] = torch.zeros(num_faces*10,1,150,150):float()
	-- test['Y'] = torch.zeros(num_faces*10)
	tr_id = 1
	tst_id = 1
	for i=1,num_faces do
		local batch = load_batch(i, MODE_TRAINING)
		for cnt=1,25 do
			-- if cnt <= 40 then
				training['X'][tr_id] = batch[cnt]
				training['Y'][tr_id] = i
				tr_id = tr_id + 1
			-- else
			-- 	test['X'][tst_id] = batch[cnt]
			-- 	test['Y'][tst_id] = i
			-- 	tst_id = tst_id + 1
			-- end
		end
		print('face:', i,'/',num_faces)
	end
	torch.save('DATASET/FINETUNE/training.t7', training)
	-- torch.save('DATASET/FINETUNE/test.t7', test)
else
	print('Loading cached data ...')
	training = torch.load('DATASET/FINETUNE/training.t7')
	-- test = torch.load('DATASET/FINETUNE/test.t7')	
end

num_training = (#training['Y'])[1]
-- num_test = (#test['Y'])[1]

classes = {}
for i=1,num_faces do
	classes[i] = i
end

classifier = nn.Sequential()
for i=1,10 do
	classifier:add(encoder.modules[i])
end
-- classifier:add(model:get(2))
-- classifier:add(nn.Dropout(0.5))
classifier_new = nn.Sequential()
classifier_new:add(nn.Linear(2250, 50))
classifier_new:add(nn.ReLU())
-- classifier:add(nn.Linear(2250, #classes))
classifier_new:add(nn.Linear(50, #classes))
classifier:add(classifier_new)

classifier:add(nn.LogSoftMax())
classifier:cuda()
-- print(classifier)

-- res = classifier:forward(load_batch(1, MODE_TEST):cuda())
-- print(res:size())


-- retrieve parameters and gradients
parameters,gradParameters = classifier_new:getParameters()

criterion = nn.ClassNLLCriterion()

criterion:cuda()



-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
-- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

print('num_train_batches:', num_train_batches)
batchSize = 25

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
	  
	  local inputs = torch.zeros(batchSize, 1, 150,150)
	  local targets = torch.zeros(batchSize, 1)
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
	        -- print('target:', targets[i][1], ' pred:', ind[1])
	        if targets[i][1] ~= ind[1] then
	        	tr_err = tr_err + 1
	        end
	      end

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
   for t = 1,num_test_batches do
      -- disp progress
      xlua.progress(t, num_test_batches)

	  local inputs = torch.zeros(batchSize, 1, 150,150)
	  local targets = torch.zeros(batchSize, 1)
	  for bid=1,50 do
	  	jj = (t-1)*batchSize +bid
	  	inputs[bid] = test['X'][jj]
	  	targets[bid] = test['Y'][jj]
	  end    
	  inputs = inputs:cuda()
	  targets = targets:cuda()
      -- -- create mini batch
      -- local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      -- local targets = torch.Tensor(opt.batchSize)

      -- test samples
      local preds = classifier:forward(inputs)
      
      for i = 1,batchSize do
        -- confusion:add(preds[i]:float(), targets[i][1])
        predexp = torch.exp(preds[i]:float())
        maxv,ind = torch.max(predexp,1)
        -- print('target:', targets[i][1], ' pred:', ind[1])
        if targets[i][1] ~= ind[1] then
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
   print('<TEST> Errors:', tst_err)--confusion.totalValid*100)
   testLogger:add{['% mean class error (test set)'] = tst_err}--confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   trainf()
   -- testf()

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      -- testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      -- testLogger:plot()
   end
end




--]]