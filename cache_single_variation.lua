
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
num_train_batches = 3000
num_test_batches =  350 --TOTALFACES-num_train_batches

SET_NAME = "LIGHT_AZ_VARIED"
DATASET_DIR = '/om/user/tejask/facemachine/CNN_DATASET/'..SET_NAME
OUTPUT_DIR = 'CNN_DATASET'

function cache(dirname, mode, batch_index)
  collectgarbage()
  os.execute('mkdir -p '..OUTPUT_DIR..'/th_'..SET_NAME..'/'..mode)

  batch = torch.zeros(bsize,1,imwidth,imwidth)

  i = 1
  for filename in lfs.dir(DATASET_DIR..'/' .. dirname) do
    if i > bsize then
      break
    end
    if string.sub(filename, string.len(filename) - 3, string.len(filename)) == ".png" then
      local im_tmp = image.load(DATASET_DIR..'/' .. dirname .. '/' .. filename)

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

  torch.save(OUTPUT_DIR..'/th_'..SET_NAME..'/'..mode..'/batch' .. batch_index, batch:float())
end

local batch_index = 1
local mode = 'FT_training'
for dirname in lfs.dir(DATASET_DIR) do
  if batch_index > num_train_batches then
    mode = 'FT_test'
    batch_index = 1
  end
  if lfs.attributes(DATASET_DIR .. '/' .. dirname).mode == 'directory' then
    cache(dirname, mode, batch_index)
    print(batch_index)
    batch_index = batch_index + 1
  end
end


-- batch_index = 1
-- for dirname in lfs.dir(DATASET_DIR) do
--   if batch_index > num_test_batches then
--     break
--   end
--   if lfs.attributes(DATASET_DIR .. '/' .. dirname).mode == 'directory' then
--     cache(dirname, 'FT_test', batch_index)
--     print(batch_index)
--     batch_index = batch_index + 1
--   end
-- end
--testing speed of loading batch
-- require 'sys'
-- for rep = 1,20 do
--  sys.tic()
--  batch = torch.load('DATASET/th_lstraining/batch' .. rep)
--  print(sys.toc())
-- end
