local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function KLDCriterion:updateOutput(input, target)
    -- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    self.term1 = self.term1 or input[1].new()
    self.term2 = self.term2 or input[2].new()
    self.term3 = self.term3 or input[2].new()

    self.term1:resizeAs(input[1])
    self.term2:resizeAs(input[2])
    self.term3:resizeAs(input[2])

    -- sigma^2
    self.term1:copy(input[2]):exp()

    -- mu^2
    self.term2:copy(input[1]):pow(2)

    -- 1 + log(sigma^2)
    self.term3:fill(1):add(input[2])

    -- 1 + log(sigma^2) - mu^2
    self.term3:add(-1,self.term2)

    -- 1 + log(sigma^2) - mu^2 - sigma^2
    self.term3:add(-1,self.term1)

    if self.sizeAverage then
      self.term3:div(input[1]:nElement())
   end

    self.output = 0.5 * self.term3:sum()

    return self.output
end

function KLDCriterion:updateGradInput(input, target)
    self.gradInput = {}

    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(input[1])
    self.gradInput[1]:copy(input[1]):mul(-1)


    self.term = self.term or input[2].new()
    self.term:resizeAs(input[2])
    self.term:copy(input[2])

    -- (- sigma^2 + 1) * 0.5
    self.gradInput[2] = self.term:exp():mul(-1):add(1):mul(0.5)

    if self.sizeAverage then
        self.gradInput:div(input[1]:nElement())
    end

    return self.gradInput
end
