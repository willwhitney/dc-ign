require 'modules/UnPooling'

--Global variables for config
bsize = 50
imwidth = 150

TOTALFACES = 5230
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

-- config = {
--     learningRate = -0.0005,
--     momentumDecay = 0.1,
--     updateDecay = 0.01
-- }

function init_network2_150()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 120--200
  feature_maps = 96
  colorchannels = 1
  dim_hidden = 120--120
  feature_maps = 96 --96

  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))


  encoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.Reshape((feature_maps/4)*15*15))

  local z = nn.ConcatTable()
  z:add(nn.LinearCR((feature_maps/4)*15*15, dim_hidden))
  z:add(nn.LinearCR((feature_maps/4)*15*15, dim_hidden))
  encoder:add(z)

  decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, (feature_maps/4)*15*15 ))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/4),15,15))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps/4,feature_maps/2, 7, 7))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,7,7))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,7,7))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps,1,7,7))
  decoder:add(nn.Sigmoid())

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end

function init_network2_150_mv(dim_hidden, feature_maps)
 -- Model Specific parameters
  filter_size = 5
  colorchannels = 1

  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))


  encoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.Reshape((feature_maps/4)*15*15))

  local z = nn.ConcatTable()

  local mu = nn.Sequential()
    mu:add(nn.LinearCR((feature_maps/4)*15*15, dim_hidden))
    mu:add(nn.SelectiveGradientFilter())
    mu:add(nn.SelectiveOutputClamp())
  z:add(mu)

  local sigma = nn.Sequential()
    sigma:add(nn.LinearCR((feature_maps/4)*15*15, dim_hidden))
    sigma:add(nn.SelectiveGradientFilter())
    sigma:add(nn.SelectiveOutputClamp())
  z:add(sigma)

  encoder:add(z)

  decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, (feature_maps/4)*15*15 ))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/4),15,15))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps/4,feature_maps/2, 7, 7))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,7,7))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,7,7))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps,1,7,7))
  decoder:add(nn.Sigmoid())

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end

--[[ Various other models which either didn't work at all or didn't work well

function init_network1()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 30*3
  input_size = 32*2
  pad1 = 2
  pad2 = 2
  colorchannels = 3
  total_output_size = colorchannels * input_size ^ 2
  feature_maps = 16*2*2
  hidden_dec = 25*2
  map_size = 16*2
  factor = 2
  encoder = nn.Sequential()
  encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))
  encoder:add(nn.Reshape(feature_maps * map_size * map_size))
  local z = nn.ConcatTable()
  z:add(nn.LinearCR(feature_maps * map_size * map_size, dim_hidden))
  z:add(nn.LinearCR(feature_maps * map_size * map_size, dim_hidden))
  encoder:add(z)
  local decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size * map_size))
  decoder:add(nn.Threshold(0,1e-6))
  --Reshape and transpose in order to upscale
  decoder:add(nn.Reshape(bsize, feature_maps, map_size, map_size))
  decoder:add(nn.Transpose({2,3},{3,4}))
  --Reshape and compute upscale with hidden dimensions
  decoder:add(nn.Reshape(map_size * map_size * bsize, feature_maps))
  decoder:add(nn.LinearCR(feature_maps,hidden_dec))
  decoder:add(nn.Threshold(0,1e-6))
  decoder:add(nn.LinearCR(hidden_dec,colorchannels*factor*factor))
  decoder:add(nn.Sigmoid())
  decoder:add(nn.Reshape(bsize,1,input_size,input_size))

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end



function init_network2()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 60
  feature_maps = 32
  colorchannels = 1

  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.Reshape((feature_maps/2)*13*13))

  local z = nn.ConcatTable()
  z:add(nn.LinearCR((feature_maps/2)*13*13, dim_hidden))
  z:add(nn.LinearCR((feature_maps/2)*13*13, dim_hidden))
  encoder:add(z)

  decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, (feature_maps/2)*13*13 ))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/2),13,13))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,filter_size,filter_size))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,2*filter_size,2*filter_size))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(feature_maps/2,1,7,7))
  decoder:add(nn.Sigmoid())

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end

function init_network2_full_150()
  -- Model Specific parameters
  batchSize = 50
  filter_size = 5
  filter_size_2 = 5
  stride = 2
  stride_2 = 2
  dim_hidden = 25
  input_size = 150
  pad1 = 2 --NB new size must be divisible with filtersize
  pad2 = 2
  pad2_1 = 2
  pad2_2 = 2
  total_output_size = 1 * input_size ^ 2
  feature_maps = 15
  feature_maps_2 = feature_maps*2
  map_size = 16
  map_size_2 = 37
  factor = 2
  factor_2 = 2
  colorchannels = 1
  --hidden_dec should be in order of: featuremaps * filtersize^2 / (16+factor^2)
  hidden_dec = 50
  hidden_dec_2 = 50
  --layer1
  encoder = nn.Sequential()
  encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
  encoder:add(nn.SpatialConvolutionMM(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))
  --layer2
  encoder:add(nn.SpatialZeroPadding(pad2_1,pad2_2,pad2_1,pad2_2))
  encoder:add(nn.SpatialConvolutionMM(feature_maps,feature_maps_2,filter_size_2,filter_size_2))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))
  encoder:add(nn.Reshape(feature_maps_2 * map_size_2 * map_size_2))
  local z = nn.ConcatTable()
  z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))
  z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))
  encoder:add(z)

  local decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, feature_maps_2 * map_size_2 * map_size_2))
  decoder:add(nn.Threshold(0,1e-6))
  decoder:add(nn.Reshape(batchSize, feature_maps_2, map_size_2, map_size_2))
  decoder:add(nn.Transpose({2,3},{3,4}))
  --layer2
  decoder:add(nn.Reshape(batchSize,feature_maps_2,map_size_2,map_size_2))
  decoder:add(nn.Transpose({2,3},{3,4}))
  decoder:add(nn.Reshape(map_size_2*map_size_2*batchSize,feature_maps_2))
  decoder:add(nn.LinearCR(feature_maps_2,hidden_dec_2))
  decoder:add(nn.Threshold(0,1e-6))
  decoder:add(nn.LinearCR(hidden_dec_2,feature_maps*factor*factor))
  decoder:add(nn.Threshold(0,1e-6))
  --layer1
  decoder:add(nn.LinearCR(feature_maps*factor*factor,hidden_dec))
  decoder:add(nn.Threshold(0,1e-6))
  decoder:add(nn.LinearCR(hidden_dec,colorchannels*factor^4))
  decoder:add(nn.Sigmoid())
  decoder:add(nn.Reshape(batchSize,total_output_size))
  -- decoder:add(nn.Reshape(batchSize, total_output_size))

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)
  return model
end

function init_network2_color_width32()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 200
  feature_maps = 64
  colorchannels = 3

  encoder = nn.Sequential()

  encoder:add(nn.Reshape(3,32,32))

  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))


  -- encoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size))
  -- encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  -- encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.Reshape((feature_maps/2)*5*5))

  local z = nn.ConcatTable()
  z:add(nn.LinearCR((feature_maps/2)*5*5, dim_hidden))
  z:add(nn.LinearCR((feature_maps/2)*5*5, dim_hidden))
  encoder:add(z)

  decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, (feature_maps/2)*12*12 ))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/2),12,12))

  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps/2,filter_size,filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,filter_size,filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,filter_size,filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialConvolution(feature_maps,colorchannels,9,9))
  decoder:add(nn.Sigmoid())

  decoder:add(nn.Reshape(3*32*32))

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end





function init_network2_color_width150()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 500
  feature_maps = 64
  colorchannels = 3

  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))


  encoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.Reshape((feature_maps/4)*15*15))

  encoder:add(nn.LinearCR((feature_maps/4)*15*15, 1024))
  encoder:add(nn.Threshold(0,1e-6))

  local z = nn.ConcatTable()
  z:add(nn.LinearCR(1024, dim_hidden))
  z:add(nn.LinearCR(1024, dim_hidden))
  encoder:add(z)

  decoder = nn.Sequential()

  decoder:add(nn.LinearCR(dim_hidden, 1024 ))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape(1, 32, 32 ))
  decoder:add(nn.SpatialConvolution(1,feature_maps/2, 9, 9))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps/2, 8, 8))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps, 5, 5))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.SpatialConvolution(feature_maps,colorchannels, 7, 7))
  decoder:add(nn.Sigmoid())

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end




function init_network3()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 60
  feature_maps = 64
  colorchannels = 1

  encoder = nn.Sequential()

  encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(nn.Reshape((feature_maps/2)*13*13))

  local z = nn.ConcatTable()
  z:add(nn.LinearCR((feature_maps/2)*13*13, dim_hidden))
  z:add(nn.LinearCR((feature_maps/2)*13*13, dim_hidden))
  encoder:add(z)

  decoder = nn.Sequential()
  decoder:add(nn.LinearCR(dim_hidden, (feature_maps/2)*13*13 ))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/2),13,13))

  -- decoder:add(nn.SpatialUpSamplingNearest(2))

  decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,filter_size,filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))


  decoder:add(nn.SpatialConvolution(feature_maps,feature_maps/4,filter_size,filter_size))
  -- decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/4)*14*14))
  decoder:add(nn.LinearCR((feature_maps/4)*14*14, 64*64))
  decoder:add(nn.Sigmoid())

  decoder:add(nn.Reshape(1, 64, 64))

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)

  --model:cuda()
  collectgarbage()
  return model
end
--]]
