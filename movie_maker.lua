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
-- network_search_str = "donatello"
network_search_str = "MV_lategradfilter_fixedindices_import_donatello_shape_bias_shape_bias_amount_400"

if true then
  -- base_directory = "/om/user/wwhitney/facegen_networks"
  base_directory = "/om/user/wwhitney/facegen"
else
  base_directory = lfs.currentdir()
end

local name_modifier_str = "light"

local jobname = network_search_str ..'_'.. os.date("%b_%d") ..'_'.. name_modifier_str
local output_path = 'movies/'..jobname
os.execute('mkdir -p '..output_path)

-- local dataset_types = {"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED", "SHAPE_VARIED"}

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

function getModel()
  -- collectgarbage()
  for network_name in lfs.dir(base_directory) do
    local network_path = base_directory .. '/' .. network_name
    if lfs.attributes(network_path).mode == 'directory' then
      if string.find(network_name, network_search_str) then
        print(network_name)
        -- cutorch.synchronize()

        local model
        if false then
          model = torch.load(network_path .. '/vxnet.net')
        else
          model = init_network2_150_mv(200, 96)
          -- cutorch.synchronize()
          -- os.execute("sleep 2")
          local parameters, gradients = model:getParameters()

          p = torch.load(network_path .. '/parameters.t7')
          parameters:copy(p)
        end
        return model
      end
    end
  end
end

function rotationTensor(length)
  local out = torch.Tensor(100)
  local rightmin = 2
  local rightmax = 29
  local leftmin = -30
  local leftmax = -4
  out[{{1, 25}}] = torch.range(rightmax - 1,rightmin,-(rightmax - rightmin) / 25)
  out[{{26,50}}] = torch.range(rightmin,rightmax - 1,(rightmax - rightmin) / 25)
  -- out[{{51,75}}] = torch.range(leftmin,leftmax - 1,(leftmax - leftmin) / 25)
  -- out[{{76,100}}] = torch.range(leftmax - 1,leftmin, -(leftmax - leftmin) / 25)
  out[{{51,75}}] = torch.range(leftmax - 1, leftmin, -(leftmax - leftmin) / 25)
  out[{{76,100}}] = torch.range(leftmin, leftmax - 1, (leftmax - leftmin) / 25)
  return out

  -- return torch.range(-49,50)
end

local image_location = '/om/user/wwhitney/facegen/movie_base_face.png'
model = getModel()

local im_tmp = image.load(image_location)
local im = torch.zeros(1,150, 150)
im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
im = im:reshape(1,1,150,150)
-- local newim = image.scale(im[1], imwidth ,imwidth)

local clamps = model:findModules('nn.SelectiveOutputClamp')
local gradFilters = model:findModules('nn.SelectiveGradientFilter')
for clampIndex = 1, #clamps do
    clamps[clampIndex].active = false
    gradFilters[clampIndex].active = false
end

local param_values = torch.sin(torch.range(0,64) * math.pi / 32) * 20
-- local param_values = rotationTensor()

local timesteps = param_values:size()[1]
local rendered = model:forward(im:cuda())

local orig_repam_out = model:get(2).output:double()
local decoder = model:get(3)

local repam_out = torch.Tensor(timesteps, 200)

for i = 1, timesteps do
  repam_out[i] = orig_repam_out[1]:clone()
end



repam_out[{{}, 3}] = param_values
local output = decoder:forward(repam_out:cuda())


local images = {}
local image_row = {}
for frameindex = 1, output:size()[1] do
  local frame = output[frameindex]:double()
  image.save(output_path..'/'..name_modifier_str ..'_'..tostring(frameindex)..'.png', frame)
  table.insert(image_row, frame)
end
table.insert(images, image_row)

saveImageGrid(output_path..'/'..name_modifier_str..'.png', images)
print("done")















