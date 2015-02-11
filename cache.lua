
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
imwidth = 64

TOTALFACES = 5230
num_train_batches = 5000
num_test_batches =  TOTALFACES-num_train_batches

function cache(id, mode)
    collectgarbage()
    local batch = torch.zeros(bsize,1,imwidth,imwidth)  
    for i=1,bsize do
      local im_tmp = image.load('DATASET/' .. mode .. '/face_' .. id .. '/' .. i .. '.png')
      im = torch.zeros(1,150, 150)
      im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
      newim = image.scale(im[1], imwidth ,imwidth)
      batch[i]=newim
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
