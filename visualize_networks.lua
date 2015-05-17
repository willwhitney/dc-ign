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
require 'modules/KLDCriterion'
require 'modules/LinearCR'
require 'modules/Reparametrize'
require 'cutorch'
require 'cunn'
require 'optim'
require 'modules/GaussianCriterion'
require 'testf'
require 'utils'
require 'config'
require 'modules/SelectiveGradientFilter'
require 'modules/SelectiveOutputClamp'
require 'lfs'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Visualize via reconstruction or generalization with params.')
cmd:text()
cmd:text('Options')
cmd:text('Change these options:')
cmd:option('--search_str', 'invariance_scaled', 'networks whose names contain this string will be rendered')
cmd:option('--base_dir',   'networks',          'absolute or relative path to networks')
cmd:option('--name_mod',   'sweep_pm_20',       'suffix to give to jobname')
cmd:option('--data_loc',   'data/faces/batch1', 'data location for generalization images')
cmd:option('--imwidth',            150, 'width (and height) of images')
cmd:option('--num_steps',            6,         'use code specific param upper/lower bounds')
cmd:option('--lower_bound',        -20,         'lower bound for params')
cmd:option('--upper_bound',         20,         'upper bound for params')
cmd:option('--custom_bounds',    false,         'use special code specific upper/lower bounds')
cmd:option('--reconstruct',      false,         'do reconstruction visualization')
cmd:option('--generalize',       false,         'do generalization visualization')
opt = cmd:parse(arg)

network_search_str = opt.search_str
base_directory = opt.base_dir
name_modifier_str = opt.name_mod
imwidth = opt.imwidth
num_steps = opt.num_steps
lower_bound = opt.lower_bound
upper_bound = opt.upper_bound
custom_bounds = opt.custom_bounds

local jobname = network_search_str ..'_'.. os.date("%b_%d_%H_%M") ..'_'.. name_modifier_str

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
  print("Saving " .. #images .. " image rows to " .. filepath)
  if images ~= nil and images[1] ~= nil then
    image_width = imwidth
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
if opt.reconstruct then
  local reconstruction_path = 'renderings/'..jobname..'/reconstruction'
  os.execute('mkdir -p '..reconstruction_path)
  local id=1
  for network_name in lfs.dir(base_directory) do
    local network_path = base_directory .. '/' .. network_name
    -- print("SO:", network_path)
    if lfs.attributes(network_path).mode == 'directory' then
      if string.find(network_name, network_search_str) then
        print(network_name)
        local images = {}
        for _, dataset_type in ipairs(dataset_types) do
          -- note: not sure what is supposed to be in these tmp directories to trigger this
          if lfs.attributes(base_directory ..'/tmp/'..network_name.."/"..dataset_type) ~= nil then
            local last_epoch = lastepochnum(base_directory ..'/tmp/'..network_name.."/"..dataset_type)

            local reconstruction_gt = torch.load('CNN_DATASET/th_'..dataset_type..'/FT_test/batch' .. id)
            local preds = torch.load(base_directory ..'/tmp/'..network_name.."/"..dataset_type.."/epoch_"..last_epoch..'/preds' ..id)


            for i=1, preds:size()[1] do
              local image_row = {}
              local gt_img = reconstruction_gt[i]
              local inf_img = preds[i]
              table.insert(image_row, gt_img)
              table.insert(image_row, inf_img)
              -- image.save('reconstruction_gt_'..tostring(i)..'.png', gt_img)
              -- image.save(tostring(i)..'.png', gt_img)
              table.insert(images, image_row)
            end
          end

        end
        saveImageGrid(reconstruction_path..'/'..network_name..'.png', images)

      end
    end
  end
end

---------------------- GENERALIZATION ----------------------
--original data_loc: '/home/tejas/Documents/MIT/facegen/DATASET/CNN_DATASET/AZ_VARIED/face_4
local data_location = opt.data_loc
local bsize = 20

if opt.generalize then
  local generalization_path = 'renderings/'..jobname..'/generalization'
  os.execute('mkdir -p '..generalization_path)

  skipnum = 0
  network_index = 1
  for network_name in lfs.dir(base_directory) do
    local network_path = base_directory .. '/' .. network_name
    if lfs.attributes(network_path).mode == 'directory' then
      if string.find(network_name, network_search_str) then
        if network_index <= skipnum then
          network_index = network_index + 1
        else
          if lfs.attributes(network_path .. '/vxnet.net') ~= nil then
            print(network_name)
            collectgarbage()
            cutorch.synchronize()

            local model
            if true then
              model = torch.load(network_path .. '/vxnet.net')
            else
              model = init_network2_150_mv(200, 96)
              local parameters, gradients = model:getParameters()

              p = torch.load(network_path .. '/parameters.t7')
              parameters:copy(p)
            end

            local images = {}

            local batch = torch.zeros(bsize,1,imwidth,imwidth)
            local image_index = 1
            for filename in lfs.dir(data_location) do
              print("filename: " .. filename)
              if string.sub(filename, string.len(filename) - 3, string.len(filename)) == ".png" then
                local im_tmp = image.load(data_location .. "/" .. filename)
                print("loading: " .. data_location .. "/" .. filename)
                local im = torch.zeros(1,imwidth, imwidth)
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

            custom_params_table = {
              {-4, 5},
              {0.5, 6},
              {-1.5, 4}
            }
            for dataset_index, dataset_type in ipairs({"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"}) do

              local rendered = model:forward(batch:cuda())

              local orig_repam_out = model:get(2).output:double()
              local decoder = model:get(3)

              --changing value of input to decoder (only first image for now)
              for id = 1,8 do

                local gt_image = batch[id]:double()
                local inf_image = rendered[id]:double()

                local image_list = {}
                table.insert(image_list, gt_image)
                table.insert(image_list, inf_image)

                step_low = lower_bound
                step_high = upper_bound
                if(custom_bounds) then
                  step_low = custom_params_table[dataset_index][1]
                  step_high = custom_params_table[dataset_index][2]
                end
                for step = 1,num_steps do
                  i = step_low + (((step-1) * (step_high - step_low)) / (num_steps - 1))
                  -- print("Will render: " .. i .. " = " .. step_low .. "," .. step_high .. "," .. step)

                  local indxs = torch.Tensor({dataset_index})--torch.randperm(200)[{{1,10}}]
                  local repam_out = orig_repam_out:clone()

                  for j= 1, indxs:size()[1] do
                    -- local param_setting = -15 --torch.uniform(-4,4)
                    -- if i == 1 then
                    --     repam_out[id][indxs[j]] = param_setting --torch.normal(0,1)--repam_out[id][indxs[j]] + torch.uniform(-5,5) --torch.uniform(-5,5)
                    -- else
                    --     repam_out[id][indxs[j]] = -param_setting
                    -- end
                    repam_out[id][indxs[j]] = i
                  end

                  local inf_changed_image = decoder:forward(repam_out:cuda())
                  local inf_changed_image = inf_changed_image[id]:double()
                  table.insert(image_list, inf_changed_image)
                end
                table.insert(images, image_list)
              end
              saveImageGrid(generalization_path..'/'..network_name..'.png', images)

            end
            network_index = network_index + 1
          end
        end
      end
    end
  end
end

print("done")











