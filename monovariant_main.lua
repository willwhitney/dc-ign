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
require 'SelectiveOutputClamp'
require 'SelectiveGradientFilter'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a network to store particular information in particular nodes.')
cmd:text()
cmd:text('Options')

cmd:text('Change these options:')
cmd:option('--save',              'default',  'where to save this network and results [DO NOT LEAVE DEFAULT]')
cmd:option('--import',            'default',  'the containing folder of the network to load in. does nothing with `no_load`')
cmd:option('--datasetdir',        'CNN_DATASET',  'dataset source directory')
cmd:option('--no_load',           false,      'do not load in an existing network')
cmd:option('--shape_bias',        false,      'use more training samples from the shape set')
cmd:option('--shape_bias_amount', 15,         'the ratio of extra samples from shape set. does nothing without `shape_bias`')
cmd:option('--dim_hidden',        200,        'dimension of the representation layer')
cmd:option('--feature_maps',      96,         'number of feature maps')
cmd:option('--learning_rate',     -0.0005,    'learning rate for the network')
cmd:option('--momentum_decay',    0.1,        'decay rate for momentum in rmsprop')


cmd:text()
cmd:text()

cmd:text("Probably don't change these:")
cmd:option('--threads',2,'how many threads to use in torch')
cmd:option('--num_train_batches',5000,'number of batches to train with per epoch')
cmd:option('--num_train_batches_per_type',3000,'number of available train batches of each data type')
cmd:option('--num_test_batches',1400,'number of batches to test with')
cmd:option('--num_test_batches_per_type',350,'number of available test batches of each type')
cmd:option('--bsize',20,'number of samples per batch')

cmd:text()

opt = cmd:parse(arg)

config = {
    learningRate = opt.learning_rate,
    momentumDecay = opt.momentum_decay,
    updateDecay = 0.01
}

-- opt = {
--   threads = 2,
--   save = "MV_lr_0_0015_F96_H120_lr0_0005",
--   import = "F96_H120_lr0_0005",
--   -- dataset_name = "AZ_VARIED",
--   -- free_param_index = 1,
--   num_train_batches = 5000, -- 1050
--   num_train_batches_per_type = 3000, --30
--   num_test_batches = 1400, --90
--   num_test_batches_per_type = 350, --30
--   bsize = 20,

--   -- shape_bias = false,
--   -- shape_bias_amount = 120,
--   -- load = true
-- }

torch.setnumthreads(opt.threads)

os.execute('mkdir ' .. opt.save)

local f = assert(io.open(opt.save .. '/cmd_options.txt', 'w'))
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

MODE_TRAINING = "FT_training"
MODE_TEST = "FT_test"


model = init_network2_150_mv(opt.dim_hidden, opt.feature_maps)
-- model = init_network2_150_mv_addLinear(model)


criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

criterion:cuda()
KLD:cuda()
model:cuda()

parameters, gradients = model:getParameters()
print('Num before', #parameters)

if not opt.no_load then
  -- load all the values from the network stored in opt.import
  lowerboundlist = torch.load(opt.import .. '/lowerbound.t7')
  lowerbound_test_list = torch.load(opt.import .. '/lowerbound_test.t7')
  state = torch.load(opt.import .. '/state.t7')
  p = torch.load(opt.import .. '/parameters.t7')
  print('Loaded p size:', #p)
  parameters:copy(p)
  epoch = lowerboundlist:size(1)
  config = torch.load(opt.import .. '/config.t7')
else
  epoch = 1
end

-- only add in the clamps if they're not already there
-- if #model:findModules('nn.SelectiveGradientFilter') == 0 then
--   -- now clamp the gradients and outputs
--   -- this puts in, after the Reparametrize, a SelectiveGradientFilter
--   -- and a SelectiveOutputClamp.
--   -- Doing this after the Reparam means we only need one layer of these
--   -- is the result still correct?
--   clamp = nn.SelectiveOutputClamp()
--   gradFilter = nn.SelectiveGradientFilter()

--     clamp:cuda()
--     gradFilter:cuda()

--   model:insert(clamp, 3)
--   model:insert(gradFilter, 3)
-- else
clamps = model:findModules('nn.SelectiveOutputClamp')
gradFilters = model:findModules('nn.SelectiveGradientFilter')
-- end

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
reconstruction = 0

while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    print("about to start a batch")

    for i = 1, opt.num_train_batches do
        xlua.progress(i, opt.num_train_batches)

        --Prepare Batch
        -- local batch = load_mv_batch(i, opt.dataset_name, MODE_TRAINING)

        -- load_random_mv_batch returns two things:
        -- 1. the batch itself
        -- 2. the type of batch it has selected (AZ, EL, LIGHT_AZ)

        if opt.shape_bias then
          batch, dataset_type = load_random_mv_shape_bias_batch(MODE_TRAINING)
        else
          batch, dataset_type = load_random_mv_batch(MODE_TRAINING)
        end

        -- clear the clamp and gradFilter's existing state
        -- (otherwise the clamp will still be stuck on the last output)
        -- clamp:reset()
        -- gradFilter:reset()

        -- set the clamp and gradient passthroughs
        for clampIndex = 1, #clamps do
          -- if dataset_type == 1 then
          --   clamps[clampIndex]:setPassthroughIndices({1,2})
          --   gradFilters[clampIndex]:setPassthroughIndices({1,2})
          -- elseif dataset_type == 2 then
          --   clamps[clampIndex]:setPassthroughIndices({3,4})
          --   gradFilters[clampIndex]:setPassthroughIndices({3,4})
          -- elseif dataset_type == 3 then
          --   clamps[clampIndex]:setPassthroughIndices({5,6})
          --   gradFilters[clampIndex]:setPassthroughIndices({5,6})
          -- elseif dataset_type == 4 then
          --   clamps[clampIndex]:setPassthroughIndices({7,120})
          --   gradFilters[clampIndex]:setPassthroughIndices({7,120})
          -- end
          if dataset_type == 1 then
            clamps[clampIndex]:setPassthroughIndices(1)
            gradFilters[clampIndex]:setPassthroughIndices(1)
          elseif dataset_type == 2 then
            clamps[clampIndex]:setPassthroughIndices(2)
            gradFilters[clampIndex]:setPassthroughIndices(2)
          elseif dataset_type == 3 then
            clamps[clampIndex]:setPassthroughIndices(3)
            gradFilters[clampIndex]:setPassthroughIndices(3)
          elseif dataset_type == 4 then
            clamps[clampIndex]:setPassthroughIndices({4,120})
            gradFilters[clampIndex]:setPassthroughIndices({4,120})
          end

          clamps[clampIndex].active = true
          gradFilters[clampIndex].active = true
        end

        batch = batch:cuda()

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

    print("\nEpoch: " .. epoch ..
          " Lowerbound: " .. lowerbound/opt.num_train_batches ..
          " time: " .. sys.clock() - time)

    --Keep track of the lowerbound over time
    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist
                                  ,torch.Tensor(1,1):fill(lowerbound/opt.num_train_batches)
                                  ,1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/opt.num_train_batches)
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

    lowerbound_test = testf_MV(false)
    -- Compute the lowerbound of the test set and save it
    if true then--epoch % 2 == 0 then
        -- lowerbound_test = getLowerbound(testData.data)

         if lowerbound_test_list then
            lowerbound_test_list = torch.cat(lowerbound_test_list
                                            ,torch.Tensor(1,1):fill(lowerbound_test/opt.num_test_batches)
                                            ,1)
        else
            lowerbound_test_list = torch.Tensor(1,1):fill(lowerbound_test/opt.num_test_batches)
        end

        print('testlowerbound = ' .. lowerbound_test/opt.num_test_batches)

        --Save everything to be able to restart later
        torch.save(opt.save .. '/parameters.t7', parameters)
        torch.save(opt.save .. '/state.t7', state)
        torch.save(opt.save .. '/lowerbound.t7', torch.Tensor(lowerboundlist))
        torch.save(opt.save .. '/lowerbound_test.t7', torch.Tensor(lowerbound_test_list))
        torch.save(opt.save .. '/config.t7', config)
    end

   -- plot errors
   if false then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
end
--]]