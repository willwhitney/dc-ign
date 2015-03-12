local UnPooling, parent = torch.class('nn.UnPooling', 'nn.Module')

require 'sys'
require 'cutorch'

function UnPooling:__init(s)
   parent.__init(self)

   self.scale = s

   self.indices = torch.Tensor()
end

function UnPooling:updateOutput(input)
   input = input:float()
   local bsize = input:size()[1]
   local dim  = input:size()[3]
   local sdim = dim*self.scale
   
   self.output = torch.zeros(bsize, input:size()[2] , sdim, sdim )
   
   local ii,jj,kk; ii=1;jj=1;kk=1;

   self.mapping = {} -- store non-zero mappings for gradient calc

   for i=1,sdim,self.scale do
      jj = 1;
      for j=1,sdim,self.scale do
         self.output[{{},{},i,j}] = input[{{},{}, ii,jj}]
         self.mapping[ii ..jj] = {i,j}
         jj = jj + 1;
      end
      ii = ii + 1;
   end

   self.output = self.output:cuda()
   return self.output
end

function UnPooling:updateGradInput(input, gradOutput)

   gradOutput = gradOutput:float()
   input = input:float()

   local dim  = input:size()[3]
   
   self.gradInput = torch.zeros(bsize, input:size()[2], dim, dim)

   for ii=1,dim do
      for jj=1,dim do
         local t = self.mapping[ii .. jj]
         i = t[1]; j = t[2];
         self.gradInput[{{},{},ii,jj}] = gradOutput[{{},{}, i,j}]   
      end
   end

   self.gradInput = self.gradInput:cuda()
   return self.gradInput
end

--[[
function UnPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
--]]
