-- Basic Usage (from scratch): th main.lua --no_load
-- Ideally preload model trained using main.lua

require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'rmsprop'

require 'modules/KLDCriterion'
require 'modules/LinearCR'
require 'modules/Reparametrize'
require 'modules/SelectiveOutputClamp'
require 'modules/SelectiveGradientFilter'

require 'cutorch'
require 'cunn'
require 'optim'
require 'testf'
require 'utils'
require 'config'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a network to store particular information in particular nodes.')
cmd:text()
cmd:text('Options')

cmd:text('Change these options:')
cmd:option('--import',            'default',      'the containing folder of the network to load in. does nothing with `no_load`')
cmd:option('--networks_dir',      'networks',     'the directory to save the resulting networks in')
cmd:option('--name',              'default',      'the name for this network. used for saving the network and results')
cmd:option('--datasetdir',        'DATASET',      'dataset source directory')

cmd:option('--dim_hidden',        200,            'dimension of the representation layer')
cmd:option('--feature_maps',      96,             'number of feature maps')

cmd:option('--force_invariance',  false,          'propagate error equal to change in outputs corresponding to fixed variables')
cmd:option('--invariance_strength',0,             'multiplier for the invariance error signal')

cmd:option('--no_load',           false,          'do not load in an existing network')
cmd:option('--shape_bias',        false,          'use more training samples from the shape set')
cmd:option('--shape_bias_amount', 15,             'the ratio of extra samples from shape set. does nothing without `shape_bias`')

cmd:option('--learning_rate',     -0.0005,        'learning rate for the network')
cmd:option('--momentum_decay',    0.1,            'decay rate for momentum in rmsprop')

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
opt.save = paths.concat(opt.networks_dir, opt.name)
os.execute('mkdir ' .. opt.save)

config = {
    learningRate = opt.learning_rate,
    momentumDecay = opt.momentum_decay,
    updateDecay = 0.01
  }

torch.setnumthreads(opt.threads)

-- log out the options used for creating this network to a file in the save directory.
-- super useful when you're moving folders around so you don't lose track of things.
local f = assert(io.open(opt.save .. '/cmd_options.txt', 'w'))
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

MODE_TRAINING = "FT_training"
MODE_TEST = "FT_test"


model = init_network2_150_mv(opt.dim_hidden, opt.feature_maps)


criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

criterion:cuda()
KLD:cuda()
model:cuda()
cutorch.synchronize()

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
  epoch = 0
end

clamps = model:findModules('nn.SelectiveOutputClamp')
gradFilters = model:findModules('nn.SelectiveGradientFilter')

for clampIndex = 1, #clamps do
  gradFilters[clampIndex].force_invariance  = opt.force_invariance
  gradFilters[clampIndex].invariance_strength  = opt.invariance_strength
end

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

    -- set the clamp and gradient passthroughs
    for clampIndex = 1, #clamps do
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
        clamps[clampIndex]:setPassthroughIndices({4,opt.dim_hidden})
        gradFilters[clampIndex]:setPassthroughIndices({4,opt.dim_hidden})
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
      local encoder_output = model:get(1).output

      local KLDerr = KLD:forward(encoder_output, target)
      local dKLD_dw = KLD:backward(encoder_output, target)

      encoder:backward(batch,dKLD_dw)

      local lowerbound = err  + KLDerr

      if opt.verbose then
        print("BCE",err/batch:size(1))
        print("KLD", KLDerr/batch:size(1))
        print("lowerbound", lowerbound/batch:size(1))
      end

      return lowerbound, gradients
    end -- /opfunc

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


  -- save the current net
  if true then
    local filename = paths.concat(opt.save, 'vxnet.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end

    print('<trainer> saving network to '..filename)
    torch.save(filename, model)
  end


  -- Compute the lowerbound of the test set and save it
  testf_MV(false)
  if true then
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

