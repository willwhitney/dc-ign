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
require 'utils'
require 'lfs'
require 'image'

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


-- networks whose names contain this string will be rendered
network_search_str = "MV_lategradfilter_fixedindices_no_load"

if false then
  base_directory = "/om/user/wwhitney/facegen"
else
  base_directory = lfs.currentdir()
end
name_modifier_str = "pm_15"


local jobname = network_search_str ..'_'.. os.date("%b_%d") ..'_'.. name_modifier_str
local reconstruction_path = 'renderings/'..jobname..'/reconstruction'
local generalization_path = 'renderings/'..jobname..'/generalization'
os.execute('mkdir -p '..reconstruction_path)
os.execute('mkdir -p '..generalization_path)

local dataset_types = {"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED", "SHAPE_VARIED"}

function lastepochnum(path)
  local last = 0
  for epochname in lfs.dir(path) do
    if string.find(epochname, '_') then
      local underscore_index = string.find(epochname, '_')
      print(epochname)

      local num = tonumber(string.sub(epochname, underscore_index + 1, string.len(epochname)))
      if num > last then
        last = num
      end
    end
  end
  return last
end

function saveImageGrid(filepath, images)
  if images ~= nil and images[1] ~= nil then
    image_width = 150
    padding = 5
    images_across = #images[1]
    images_down = #images
    -- print(images_down, images_across)

    image_output = torch.zeros(
                      image_width * images_down + (images_down - 1) * padding,
                      image_width * images_across + (images_across - 1) * padding)
    for i, image_row in ipairs(images) do
      for j, image in ipairs(image_row) do
        y_index = j - 1
        y_location = y_index * image_width + y_index * padding
        x_index = i - 1
        x_location = (x_index) * image_width + x_index * padding
        -- print({{x_location + 1, x_location + image_width},
        --           {y_location + 1, y_location + image_width}})
        image_output[{{x_location + 1, x_location + image_width},
                  {y_location + 1, y_location + image_width}}] = image
      end
    end
    image_output = image_output:reshape(1, image_output:size()[1], image_output:size()[2])
    image.save(filepath, image_output)
  end
end


---------------------- RECONSTRUCTION ----------------------
-- local id=1
-- for network_name in lfs.dir(base_directory) do
--   local network_path = base_directory .. '/' .. network_name
--   -- print(network_path)
--   if lfs.attributes(network_path).mode == 'directory' then
--     if string.find(network_name, network_search_str) then
--       print(network_name)
--       local images = {}
--       for _, dataset_type in ipairs(dataset_types) do
--         if lfs.attributes(base_directory ..'/tmp/'..network_name.."/"..dataset_type) ~= nil then
--           local last_epoch = lastepochnum(base_directory ..'/tmp/'..network_name.."/"..dataset_type)

--           local reconstruction_gt = torch.load('CNN_DATASET/th_'..dataset_type..'/FT_test/batch' .. id)
--           local preds = torch.load(base_directory ..'/tmp/'..network_name.."/"..dataset_type.."/epoch_"..last_epoch..'/preds' ..id)


--           for i=1, preds:size()[1] do
--             local image_row = {}
--             local gt_img = reconstruction_gt[i]
--             local inf_img = preds[i]
--             table.insert(image_row, gt_img)
--             table.insert(image_row, inf_img)
--             -- image.save('reconstruction_gt_'..tostring(i)..'.png', gt_img)
--             -- image.save(tostring(i)..'.png', gt_img)
--             table.insert(images, image_row)
--           end
--         end

--       end
--       saveImageGrid(reconstruction_path..'/'..network_name..'.png', images)

--     end
--   end
-- end


---------------------- GENERALIZATION ----------------------
faceid = 4
local data_location = '/home/tejas/Documents/MIT/facegen/DATASET/CNN_DATASET/AZ_VARIED/face_' .. faceid
local bsize = 20

skipnum = 0
network_index = 1
for network_name in lfs.dir(base_directory) do
  local network_path = base_directory .. '/' .. network_name
  if lfs.attributes(network_path).mode == 'directory' then
    if string.find(network_name, network_search_str) then
      if network_index <= skipnum then
        network_index = network_index + 1
      else
        print(network_name)
        collectgarbage()
        cutorch.synchronize()

        local model
        if false then
          model = torch.load(network_path .. '/vxnet.net')
        else
          model = init_network2_150_mv(200, 96)
          local parameters, gradients = model:getParameters()

          p = torch.load(network_path .. '/parameters.t7')
          parameters:copy(p)
        end

        local images = {}

        local batch = torch.zeros(bsize,1,150,150)
        local image_index = 1
        for filename in lfs.dir(data_location) do
          if string.sub(filename, string.len(filename) - 3, string.len(filename)) == ".png" then
            local im_tmp = image.load(data_location .. "/" .. filename)
            local im = torch.zeros(1,150, 150)
            im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
            local newim = image.scale(im[1], imwidth ,imwidth)
            batch[image_index] = newim
            image_index = image_index + 1

            if image_index > bsize then
                break
            end
          end
        end

        local clamps = model:findModules('nn.SelectiveOutputClamp')
        local gradFilters = model:findModules('nn.SelectiveGradientFilter')
        for clampIndex = 1, #clamps do
            clamps[clampIndex].active = false
            gradFilters[clampIndex].active = false
        end
        for dataset_index, dataset_type in ipairs({"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"}) do

          local rendered = model:forward(batch:cuda())

          local orig_repam_out = model:get(2).output:double()
          local decoder = model:get(3)

          --changing value of input to decoder (only first image for now)
          for id = 1,5 do

            local gt_image = batch[id]:double()
            local inf_image = rendered[id]:double()

            local image_list = {}
            table.insert(image_list, gt_image)
            table.insert(image_list, inf_image)

            for i=1,2 do
            -- for i=-20, 20, 10 do
              local indxs = torch.Tensor({dataset_index})--torch.randperm(200)[{{1,10}}]
              local repam_out = orig_repam_out:clone()

              for j= 1, indxs:size()[1] do
                local param_setting = -15 --torch.uniform(-4,4)
                if i == 1 then
                    repam_out[id][indxs[j]] = param_setting --torch.normal(0,1)--repam_out[id][indxs[j]] + torch.uniform(-5,5) --torch.uniform(-5,5)
                else
                    repam_out[id][indxs[j]] = -param_setting
                end
                -- repam_out[id][indxs[j]] = i
              end

              local inf_changed_image = decoder:forward(repam_out:cuda())
              local inf_changed_image = inf_changed_image[id]:double()
              table.insert(image_list, inf_changed_image)
            end
            table.insert(images, image_list)
          end
          saveImageGrid(generalization_path..'/'..network_name.. faceid .. '.png', images)

        end
        network_index = network_index + 1
      end
    end
  end
end

print("done")











