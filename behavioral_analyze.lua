-- take a trained model and output features for all the behavioral test images
require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'rmsprop'
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
require 'image'

MODE_TEST = 'test'

model = torch.load('log_FINE_TUNE2/classifier.net')

print(model)
ENC_OUT = 50--200
COLOR = false
imwidth = 150

mean_arr = torch.zeros(96*2,ENC_OUT)
sigma_arr = torch.zeros(96*2,ENC_OUT)
ftr_arr = torch.zeros(96*2,ENC_OUT)

if COLOR then
  batch = torch.zeros(1,3,imwidth,imwidth)  
else
  batch = torch.zeros(1,1,imwidth,imwidth)  
end

ii = 1
for id=1,96 do
	for pid = 0,1 do
		fname = 'DATASET/images_wbg/' .. 'pair_' .. id .. '_' .. pid .. '.png'
		im_tmp = image.load(fname)

  	if COLOR==true then
      im = torch.zeros(3,imwidth, imwidth)
      if im_tmp:size()[2] ~= imwidth then
      	newim = image.scale(im_tmp, imwidth ,imwidth)
      else
      	newim = im_tmp
      end
      batch[1] = newim
  	else
    	im = torch.zeros(1,256, 256)
    	im[1] = im_tmp[1]*0.21 + im_tmp[2]*0.72 + im_tmp[3]*0.07
    	im = image.scale(im[1], imwidth ,imwidth)
    	im = torch.reshape(im, 1, imwidth, imwidth)
    	batch[1] = im
  	end

    res = model:forward(batch:cuda())
    
    -- mean = model.modules[1].modules[11].modules[1].output:double()
    -- sigma = model.modules[1].modules[11].modules[2].output:double()
    -- mean_arr[ii] = mean[1]
    -- sigma_arr[ii] = sigma[1]
    ftr_arr[ii] = model.modules[11].modules[2].output:double()
    print(ii)
    ii = ii + 1
	end
end


require 'mattorch'
-- mattorch.save('analyzeBehavioral/mean.mat', mean_arr)
-- mattorch.save('analyzeBehavioral/sigma.mat', sigma_arr)
mattorch.save('analyzeBehavioral/ftrs.mat', ftr_arr)


--]]







