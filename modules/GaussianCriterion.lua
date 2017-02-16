require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:__init()
   parent.__init(self)
end

function GaussianCriterion:updateOutput(input, target)
   -- Verify again for correct handling of 0.5 multiplication
   --local Gelement = torch.add(input[2],math.log(2 * math.pi)):mul(-0.5)
   --Gelement:add(-1,torch.add(target,-1,input[1]):pow(2):cdiv(torch.exp(input[2])):mul(0.5))
   --self.output = torch.sum(Gelement)

   -- Reusing memory buffers
   self._Gelement = self._Gelement or input[2].new()
   self._Gelement:resizeAs(input[2]):copy(input[2])
   self._Gelement:add(math.log(2*math.pi)):mul(-0.5)

   self._TI1 = self._TI1 or input[2].new()
   self._TI1:resizeAs(target):copy(target):add(-1, input[1])

   self._expI2 = self._expI2 or input[2].new()
   self._expI2:resizeAs(input[2]):copy(input[2]):exp()

   self._TI = self._TI or self._TI1.new()
   self._TI:resizeAs(self._TI1):copy(self._TI1)
   self._TI:pow(2):cdiv(self._expI2):mul(0.5)

   self._Gelement:add(-1, self._TI)

   self.output = self._Gelement:sum()
   return self.output
end

function GaussianCriterion:updateGradInput(input, target)
   -- Verify again for correct handling of 0.5 multiplication
   --self._gradInput[1] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]))
   --self._gradInput[2] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]):pow(2)):add(-0.5)

   -- Reusing memory buffers.
	self._gradInput = self._gradInput or {}
   self._gradInput[1] = self._gradInput[1] or input[2].new()
   self._gradInput[1]:resizeAs(input[2]):copy(input[2]):mul(-1):exp()
   -- Reuse self._TI1 from updateOutput
   self._gradInput[1]:cmul(self._TI1)
   
   self._gradInput[2] = self._gradInput[2] or input[2].new()
   self._gradInput[2]:resizeAs(input[2]):copy(input[2]):mul(-1):exp()
   self._TI1:pow(2)
   self._gradInput[2]:cmul(self._TI1):add(-0.5)

   self.gradInput = self._gradInput
   return self.gradInput
end

function GaussianCriterion:type(type, tensorCache)
   if type then
      -- updateOutput
      self._Gelement = nil
      self._TI1 = nil
      self._TI = nil
      self._expI2 = nil

      -- updateGradInput
      self._gradInput = nil
   end
   return parent.type(self, type, tensorCache)
end
