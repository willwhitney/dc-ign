-- pre-process and save data
require 'mattorch'
require 'sys'

TOTALFACES = 5231
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches-1

bsize = 50
imwidth = 150

function load_batch(id)
	collectgarbage()
  basedir = '../facemachine/CNN_DATASET/'
  local t = mattorch.load(basedir .. 'face_' .. id .. '/data' .. '.mat' )
  t = t.img:reshape(bsize, 3, imwidth, imwidth):float()/255
  t = torch.Tensor(t:size()):copy(t)
  return t
end


for i=1,num_train_batches do
 	local t = load_batch(i)
 	print(i, '/', num_train_batches)
 	torch.save('DATASET/training/batch' .. i)
end

for i=1,num_test_batches do
  local FID = i + num_train_batches
  local t = load_batch(FID)
  print(i, '/', num_test_batches)
 	torch.save('DATASET/test/batch' .. i)
end