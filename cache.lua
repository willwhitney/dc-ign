
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


bsize = 50
imwidth = 150

TOTALFACES = 5230
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

function cache(id, mode)
    collectgarbage()
    local batch = torch.zeros(bsize,3,imwidth,imwidth)  
    for i=1,bsize do
      local im = image.load('DATASET/' .. mode .. '/face_' .. id .. '/' .. i .. '.png')
      batch[i]=im
    end
    torch.save('DATASET/th_' .. mode .. '/batch' .. id, batch:float())
end

for t = 1, num_train_batches do
	cache(t, 'training')
	print(t)
end

for t = 1, num_test_batches do
	cache(t, 'test')
	print(t)
end

--testing speed of loading batch
-- require 'sys'
-- for rep = 1,20 do
-- 	sys.tic()
-- 	batch = torch.load('DATASET/th_lstraining/batch' .. rep)
-- 	print(sys.toc())
-- end
