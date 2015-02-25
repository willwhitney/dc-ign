require 'nn'
require 'SelectiveOutputClamp'
require 'SelectiveGradientFilter'


gradFilter = nn.SelectiveGradientFilter()
clamp = nn.SelectiveOutputClamp()

model = nn.Sequential()
model:add(nn.Linear(10,5))
model:add(gradFilter)
model:add(clamp)
