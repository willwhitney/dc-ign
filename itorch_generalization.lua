
-- require 'sys'
-- require 'xlua'
-- require 'torch'
-- require 'nn'
-- require 'rmsprop'
-- require 'image'
-- require 'KLDCriterion'
-- require 'LinearCR'
-- require 'Reparametrize'
-- require 'cutorch'
-- require 'cunn'
-- require 'optim'
-- require 'GaussianCriterion'
-- require 'testf'
-- require 'utils'
-- require 'config'

-- out_dir = 'RESULTS/render/'
-- MODE_TRAINING = 'training'
-- MODE_TEST = 'test'

-- fname = 'logs_init_network2_150'
-- model = torch.load(fname .. '/vxnet.net')
-- -- batch = load_batch(1,MODE_TRAINING)

-- bsize = 6
-- batch = torch.zeros(bsize,1,150,150)
-- for i=1,bsize do
--   im_tmp = image.load('DATASET/GENERALIZATION/test1/' .. i .. '.png')
--     im = torch.zeros(1,150, 150)
--     im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
--     newim = image.scale(im[1], imwidth ,imwidth)
--     batch[i] = newim
-- end


-- -- print(model)
-- rendered = model:forward(batch:cuda())

-- orig_repam_out = model:get(2).output:double()
-- decoder = model:get(3)

-- --changing value of input to decoder (only first image for now)
-- for id = 1,6 do

--   gt_image = batch[id]:double()
--   inf_image = rendered[id]:double()

--     image_list = {};
--     cnt=1
--     image_list[cnt] = gt_image;cnt=cnt+1
--     image_list[cnt] = inf_image;cnt=cnt+1

--     for i=134, 134 do
--     indxs = torch.Tensor({134})--torch.randperm(200)[{{1,10}}]
--     repam_out = orig_repam_out:clone()

--         --repam_out[{id,{i,i}}]=  10 --torch.uniform(-5,5) --repam_out[{id,{1,20}}] + torch.rand(10)*2

--         --logs_init_network2_150
--         -- [1,50] => shadow?
--         -- [24,54, 134(strong)] => light
--         -- 99 => pose?
--         --

--     for j= 1, indxs:size()[1] do
--       repam_out[id][indxs[j]] = 2--torch.uniform(-4,4) --torch.normal(0,1)--repam_out[id][indxs[j]] + torch.uniform(-5,5) --torch.uniform(-5,5)
--     end

--     inf_changed_image = decoder:forward(repam_out:cuda())
--     inf_changed_image = inf_changed_image[id]:double()
--     --image.savePNG(out_dir .. 'inf_modified_' .. i .. '_T' .. id .. '.png', inf_changed_image)
--         image_list[cnt] = inf_changed_image;cnt=cnt+1
--   end
--     --visialize
--     itorch.image(image_list)
-- end


require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'rmsprop'
require 'image'
require 'KLDCriterion'
require 'LinearCR'
require 'Reparametrize'
require 'cutorch'
require 'cunn'
require 'optim'
require 'GaussianCriterion'
require 'testf'
require 'utils'
require 'config'
require 'SelectiveGradientFilter'
require 'SelectiveOutputClamp'
require 'lfs'

out_dir = 'RESULTS/render/'
MODE_TRAINING = 'training'
MODE_TEST = 'test'

model_location = 'MV_F96_H120_lr0_0005_BACKUP1_clamped_sigma'
model = torch.load(model_location .. '/vxnet.net')
-- batch = load_batch(1,MODE_TRAINING)

data_location = 'DATASET/TRANSFORMATION_DATASET/AZ_VARIED/face_3'

bsize = 20
batch = torch.zeros(bsize,1,150,150)
i = 1
for filename in lfs.dir(data_location) do
    if string.sub(filename, string.len(filename) - 3, string.len(filename)) == ".png" then
        im_tmp = image.load(data_location .. "/" .. filename)
        im = torch.zeros(1,150, 150)
        im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
        newim = image.scale(im[1], imwidth ,imwidth)
        batch[i] = newim
        i = i + 1

        if i > bsize then
            break
        end
    end
end

clamps = model:findModules('nn.SelectiveOutputClamp')
gradFilters = model:findModules('nn.SelectiveGradientFilter')
for clampIndex = 1, #clamps do
    clamps[clampIndex].active = false
    gradFilters[clampIndex].active = false
end


-- print(model)
rendered = model:forward(batch:cuda())

orig_repam_out = model:get(2).output:double()
decoder = model:get(3)

--changing value of input to decoder (only first image for now)
for id = 1,bsize do

    gt_image = batch[id]:double()
    inf_image = rendered[id]:double()

    image_list = {};
    cnt=1
    image_list[cnt] = gt_image;cnt=cnt+1
    image_list[cnt] = inf_image;cnt=cnt+1

    for i=1, 2 do
        indxs = torch.Tensor({3})--torch.randperm(200)[{{1,10}}]
        repam_out = orig_repam_out:clone()

        --repam_out[{id,{i,i}}]=  10 --torch.uniform(-5,5) --repam_out[{id,{1,20}}] + torch.rand(10)*2

        --logs_init_network2_150
        -- [1,50] => shadow?
        -- [24,54, 134(strong)] => light
        -- 99 => pose?
        --

        for j= 1, indxs:size()[1] do
            local param_setting = -15--torch.uniform(-4,4)
            if i == 1 then
                repam_out[id][indxs[j]] = param_setting --torch.normal(0,1)--repam_out[id][indxs[j]] + torch.uniform(-5,5) --torch.uniform(-5,5)
            else
                repam_out[id][indxs[j]] = -param_setting
            end
        end

        inf_changed_image = decoder:forward(repam_out:cuda())
        inf_changed_image = inf_changed_image[id]:double()
        --image.savePNG(out_dir .. 'inf_modified_' .. i .. '_T' .. id .. '.png', inf_changed_image)
        image_list[cnt] = inf_changed_image;cnt=cnt+1
    end
    --visialize
    itorch.image(image_list)
end
