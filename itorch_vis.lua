require 'cunn'
require 'pl'
require 'paths'
require 'optim'
require 'gnuplot'
require 'math'
require 'rmsprop'
require 'cudnn'
require 'nnx'
require 'image'

-- id=4
-- preds = torch.load('tmp/preds' ..id)
-- gt = torch.load('DATASET/th_test' .. '/batch' .. id)
-- for i=1, preds:size()[1] do
--     gt_img = gt[i]
--     inf_img = preds[i]
--     itorch.image({gt_img, inf_img})
-- end

-- save_name = "MV_F96_H40_multiplied_gradient"
save_name = "MV_F96_H40_clamped_sigma"
dataset_type = "AZ_VARIED"
id=1
preds = torch.load('tmp/'..save_name.."/"..dataset_type..'/preds' ..id)
gt = torch.load('DATASET/TRANSFORMATION_DATASET/th_'..dataset_type..'/FT_test/batch' .. id)
for i=1, preds:size()[1] do
    gt_img = gt[i]
    inf_img = preds[i]
    itorch.image({gt_img, inf_img})
end



