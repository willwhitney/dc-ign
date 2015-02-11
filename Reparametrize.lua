-- Based on JoinTable module
require 'nn'

local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
end 

function Reparametrize:updateOutput(input)
    self.eps = torch.randn(input[2]:size(1),self.dimension)

    if torch.typename(input[1]) == 'torch.CudaTensor' then
        self.eps = self.eps:cuda()
        self.output = torch.CudaTensor():resizeAs(input[2]):fill(0.5)
    else
        self.output = torch.Tensor():resizeAs(input[2]):fill(0.5)
    end

    self.output:cmul(input[2]):exp():cmul(self.eps)

    -- Add the mean
    self.output:add(input[1])

    return self.output
end

function Reparametrize:updateGradInput(input, gradOutput)
    -- Derivative with respect to mean is 1
    self.gradInput[1] = gradOutput:clone()
    
    --test gradient with Jacobian
    if torch.typename(input[1]) == 'torch.CudaTensor' then
        self.gradInput[2] = torch.CudaTensor():resizeAs(input[2]):fill(0.5)
    else
        self.gradInput[2] = torch.Tensor():resizeAs(input[2]):fill(0.5)
    end

    self.gradInput[2]:cmul(input[2]):exp():mul(0.5):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
