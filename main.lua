-- Continue: th main.lua -t 1 -u 1 -n logs_init_network2_150/params.t7 -s logs_init_network2_150_run2
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
--[[
if opt.fbmat == 1 then
  mattorch = require('fb.mattorch')
else
  require 'mattorch'
end
]]
-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

opt.cuda = true

-- torch.manualSeed(1)

if opt.color == 1 then
  MODE_TRAINING = 'color_training'
  MODE_TEST = 'color_test'
else
  MODE_TRAINING = 'training'
  MODE_TEST = 'test'
end


print('IMWIDTH:', load_batch(1,MODE_TRAINING):size())

-- model = init_network2_color_width150()
-- model = init_network2_full_150()
model = init_network2_150()

function test_fw_back(model)
  -- print('IMWIDTH:', load_batch(1,MODE_TRAINING):size())
  res=model:forward(load_batch(1,MODE_TRAINING):cuda())
  print(res:size())
  -- rev=model:backward(load_batch(1,MODE_TRAINING):cuda(), load_batch(1,MODE_TRAINING):cuda())
  --  print(rev:size())
end

-- test_fw_back(model)


-- criterion = nn.MSECriterion() -- does not work well at all
-- criterion = nn.GaussianCriterion()

criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

if opt.cuda then
    criterion:cuda()
    KLD:cuda()
    model:cuda()
end

parameters, gradients = model:getParameters()
print('Num before', #parameters)

-- if opt.reuse == 1 then
--     print("Loading old parameters!")
--     -- model = torch.load(opt.network)
--     parameters = torch.load(opt.network)
--     -- parameters, gradients = model:getParameters()
-- else
--   epoch = 0
--   state = {}
-- end


if opt.reuse == 1 then
  print("Loading old weights!")
  print(opt.save)
  lowerboundlist = torch.load(opt.save .. '/lowerbound.t7')
  lowerbound_test_list = torch.load(opt.save .. '/lowerbound_test.t7')
  state = torch.load(opt.save .. '/state.t7')
  p = torch.load(opt.save .. '/parameters.t7')
  print('Loaded p size:', #p)
  parameters:copy(p)
  epoch = lowerboundlist:size(1)
  config = torch.load(opt.save .. '/config.t7')
else
  epoch = 0
  state = {}
end

print('Num of parameters:', #parameters)


testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
reconstruction = 0

while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()

    for i = 1, num_train_batches do
        xlua.progress(i, num_train_batches)

        --Prepare Batch
        local batch = load_batch(i, MODE_TRAINING)

         if opt.cuda then
            batch = batch:cuda()
        end

        --Optimization function
        local opfunc = function(x)
            collectgarbage()

            if x ~= parameters then
                parameters:copy(x)
            end

            model:zeroGradParameters()
            local f = model:forward(batch)

            local target = target or batch.new()
            target:resizeAs(f):copy(batch)

            local err = - criterion:forward(f, target)
            local df_dw = criterion:backward(f, target):mul(-1)

            model:backward(batch,df_dw)
            -- local encoder_output = model.modules[1].modules[11].output
            local encoder_output = model:get(1).output

            local KLDerr = KLD:forward(encoder_output, target)
            local dKLD_dw = KLD:backward(encoder_output, target)

            -- print(encoder_output)
            -- print(batch:size())

            encoder:backward(batch,dKLD_dw)

            local lowerbound = err  + KLDerr

            if opt.verbose then
                print("BCE",err/batch:size(1))
                print("KLD", KLDerr/batch:size(1))
                print("lowerbound", lowerbound/batch:size(1))
            end

            return lowerbound, gradients
        end

        x, batchlowerbound = rmsprop(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/num_train_batches .. " time: " .. sys.clock() - time)

    --Keep track of the lowerbound over time
    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/num_train_batches),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/num_train_batches)
    end


   -- save/log current net
   if true then --math.fmod(epoch, 2) ==0 then
     local filename = paths.concat(opt.save, 'vxnet.net')
     os.execute('mkdir -p ' .. sys.dirname(filename))
     if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
     end
     print('<trainer> saving network to '..filename)
     torch.save(filename, model)
   end

    lowerbound_test = testf(false)
    -- Compute the lowerbound of the test set and save it
    if true then--epoch % 2 == 0 then
        -- lowerbound_test = getLowerbound(testData.data)

         if lowerbound_test_list then
            lowerbound_test_list = torch.cat(lowerbound_test_list,torch.Tensor(1,1):fill(lowerbound_test/num_test_batches),1)
        else
            lowerbound_test_list = torch.Tensor(1,1):fill(lowerbound_test/num_test_batches)
        end

        print('testlowerbound = ' .. lowerbound_test/num_test_batches)

        --Save everything to be able to restart later
        torch.save(opt.save .. '/parameters.t7', parameters)
        torch.save(opt.save .. '/state.t7', state)
        torch.save(opt.save .. '/lowerbound.t7', torch.Tensor(lowerboundlist))
        torch.save(opt.save .. '/lowerbound_test.t7', torch.Tensor(lowerbound_test_list))
        torch.save(opt.save .. '/config.t7', config)
    end

   -- plot errors
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
end

--]]
