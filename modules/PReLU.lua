local PReLU, parent = torch.class('nn.PReLU', 'nn.Module')

function PReLU:__init()
  parent.__init(self)
end

function PReLU:updateOutput(input)
  self.output:copy(self.bias)
  return self.output
end

function PReLU:updateGradInput(input, gradOutput)
  self.gradInput = torch.zeros(input:size())
  return self.gradInput
end

function PReLU:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  self.gradBias:add(scale, gradOutput)
end

function PReLU:zeroGradParameters()
  self.gradBias = torch.zeros(self.gradBias:size())
end

function PReLU:updateParameters(learningRate)
  self.bias:add(-learningRate, self.gradBias)
end

function PReLU:accUpdateGradParameters(input, gradOutput, learningRate)
  local gradBias = self.gradBias
  self.gradBias = self.bias
  self:accGradParameters(input, gradOutput, -learningRate)
  self.gradBias = gradBias

end



