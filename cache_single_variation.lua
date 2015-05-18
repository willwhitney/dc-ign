
require 'pl'
require 'paths'
require 'optim'
require 'gnuplot'
require 'math'
require 'image'
require 'lfs'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Convert png batch directories into cached torch tensors.')
cmd:text()
cmd:text('Options')
cmd:text('Change these options:')
cmd:option('--bsize',                           20, 'size of each batch')
cmd:option('--imwidth',                        150, 'width (and height) of images (stick with 150 for now)')
cmd:option('--num_train_batches',             3000, 'number of training batches to convert')
cmd:option('--num_test_batches',               350, 'number of test batches to convert')
cmd:option('--set_name',               "AZ_VARIED", 'name of set')
cmd:option('--dataset_dir',  'data/eyes/AZ_VARIED', 'source locaiton of dataset')
cmd:option('--output_dir',  'data/eyes/cnn_cached', 'target locaiton of dataset')
opt = cmd:parse(arg)

bsize = opt.bsize
imwidth = opt.imwidth
network_search_str = opt.search_str
base_directory = opt.base_dir
name_modifier_str = opt.name_mod
num_train_batches = opt.num_train_batches
num_test_batches =  opt.num_test_batches
SET_NAME = opt.set_name
DATASET_DIR = opt.dataset_dir
OUTPUT_DIR = opt.output_dir

function cache(dirname, mode, batch_index)
  collectgarbage()
  os.execute('mkdir -p '..OUTPUT_DIR..'/th_'..SET_NAME..'/'..mode)

  batch = torch.zeros(bsize,1,imwidth,imwidth)

  i = 1
  for filename in lfs.dir(dirname) do
    if i > bsize then
      break
    end
    if string.sub(filename, string.len(filename) - 3, string.len(filename)) == ".png" then
      local im_tmp = image.load(dirname .. '/' .. filename)

      if COLOR==true then
        im = torch.zeros(3,imwidth, imwidth)
        if im:size()[2] ~= imwidth then
          newim = image.scale(im, imwidth ,imwidth)
        else
          newim = im
        end
      else
        im = torch.zeros(1,imwidth, imwidth)
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
local mode = 'FT_test'
for dirname in lfs.dir(DATASET_DIR) do
  if mode == 'FT_test' and batch_index > num_test_batches then
    mode = 'FT_training'
    batch_index = 1
  end
  if mode == 'FT_training' and batch_index > num_train_batches then
    break
  end
  full_dirname = DATASET_DIR .. '/' .. dirname
  if dirname:sub(0,1) ~= '.' and lfs.attributes(full_dirname).mode == 'directory' then
    cache(full_dirname, mode, batch_index)
    print(mode .. ' : ' .. batch_index)
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
