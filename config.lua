require 'UnPooling'
require 'cudnn'
--Global variables for config
bsize = 50
imwidth = 150

TOTALFACES = 5230
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

config = {
    learningRate = -0.001,
    momentumDecay = 0.1,
    updateDecay = 0.01
}


function init_network1()
 -- Model Specific parameters
  filter_size = 5
  dim_hidden = 30*3
  input_size = 32*2
  pad1 = 2
  pad2 = 2
  colorchannels = 1
  total_output_size = colorchannels * input_size ^ 2
  feature_maps = 16*2*2
  hidden_dec = 25*2
  map_size = 16*2
  factor = 2
  encoder = nn.Sequential()
  encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
  encoder:add(cudnn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(cudnn.SpatialMaxPooling(2,2,2,2))
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
  decoder:add(cudnn.Sigmoid())
  decoder:add(nn.Reshape(bsize,1,input_size,input_size))

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)
    
  model:cuda()  
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

  encoder:add(cudnn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(cudnn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(cudnn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(cudnn.SpatialMaxPooling(2,2,2,2))
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
  
  decoder:add(cudnn.SpatialConvolution(feature_maps/2,feature_maps,filter_size,filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))


  decoder:add(cudnn.SpatialConvolution(feature_maps,feature_maps/2,2*filter_size,2*filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(cudnn.SpatialConvolution(feature_maps/2,1,7,7))
  decoder:add(cudnn.Sigmoid())

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)
    
  model:cuda()  
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

  encoder:add(cudnn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
  encoder:add(cudnn.SpatialMaxPooling(2,2,2,2))
  encoder:add(nn.Threshold(0,1e-6))

  encoder:add(cudnn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size))
  encoder:add(cudnn.SpatialMaxPooling(2,2,2,2))
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
  
  decoder:add(cudnn.SpatialConvolution(feature_maps/2,feature_maps,filter_size,filter_size))
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))


  decoder:add(cudnn.SpatialConvolution(feature_maps,feature_maps/4,filter_size,filter_size))
  -- decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.Threshold(0,1e-6))

  decoder:add(nn.Reshape((feature_maps/4)*14*14))
  decoder:add(nn.LinearCR((feature_maps/4)*14*14, 64*64))
  decoder:add(cudnn.Sigmoid())

  decoder:add(nn.Reshape(1, 64, 64))

  model = nn.Sequential()
  model:add(encoder)
  model:add(nn.Reparametrize(dim_hidden))
  model:add(decoder)
    
  model:cuda()  
  collectgarbage()
  return model
end
