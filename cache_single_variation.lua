
require 'pl'
require 'paths'
require 'optim'
require 'gnuplot'
require 'math'
require 'image'
require 'lfs'


bsize = 20
imwidth = 150

-- TOTALFACES = 395
num_train_batches = 350
num_test_batches =  30 --TOTALFACES-num_train_batches

SET_NAME = "LIGHT_AZ_VARIED"

function cache(batch_index, mode)
  collectgarbage()

  batch = torch.zeros(bsize,1,imwidth,imwidth)

  i = 1
  for filename in lfs.dir('DATASET/TRANSFORMATION_DATASET/'..SET_NAME..'/face_' .. batch_index) do
    if string.sub(filename, string.len(filename) - 3, string.len(filename)) == ".png" then
      local im_tmp = image.load('DATASET/TRANSFORMATION_DATASET/'..SET_NAME..'/face_' .. batch_index .. '/' .. filename)

      if COLOR==true then
        im = torch.zeros(3,150, 150)
        if im:size()[2] ~= imwidth then
          newim = image.scale(im, imwidth ,imwidth)
        else
          newim = im
        end
      else
        im = torch.zeros(1,150, 150)
        im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
        newim = image.scale(im[1], imwidth ,imwidth)
      end
      batch[i]=newim
      i = i + 1
    end
  end

  torch.save('DATASET/TRANSFORMATION_DATASET/th_'..SET_NAME..'/'..mode..'/batch' .. batch_index, batch:float())
end

for batch_index = 1, num_train_batches do
  cache(batch_index, 'FT_training')
  print(batch_index)
end

for batch_index = 1, num_test_batches do
  cache(batch_index, 'FT_test')
  print(batch_index)
end

--testing speed of loading batch
-- require 'sys'
-- for rep = 1,20 do
--  sys.tic()
--  batch = torch.load('DATASET/th_lstraining/batch' .. rep)
--  print(sys.toc())
-- end
