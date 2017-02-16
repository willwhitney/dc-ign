require 'nn'

local SelectiveGradientFilter, parent = torch.class('nn.SelectiveGradientFilter', 'nn.Module')

--[[

The API here is a little tricky, so be careful.

When it's created, the SelectiveGradientFilter has nil passthrough.

Call `setPassthroughIndex` with the index of the data you want it to pass
through unchanged.

From then until it is reset, its gradInput for all indices other than `self.passthrough`
will be zero.

Outputs are passed through unchanged.

Gradients:
0  | <-
0  | <-
0  | <-
0  | <-
0  | <-
<-   <-     -- this is the passthrough
0  | <-
0  | <-


Outputs:
-> | ->
-> | ->
-> | ->
-> | ->
-> | ->
->   ->
-> | ->
-> | ->
--]]


function SelectiveGradientFilter:__init()
    parent.__init(self)
    self:reset()
    self.gradInput = torch.Tensor()
end

function SelectiveGradientFilter:setPassthroughIndices(indices)
    self.passthrough = indices
end

function SelectiveGradientFilter:reset()
    self.passthrough = nil
    self.active = true
    self.force_invariance = false
    self.invariance_strength = 0
end

function SelectiveGradientFilter:updateOutput(input)
    self.output = input
    return self.output
end

function SelectiveGradientFilter:updateGradInput(input, gradOutput)
    if self.active then
        if self.passthrough == nil then
            if self.force_invariance then
                self.gradInput:resizeAs(input)

                local target = torch.mean(input, 1) -- mean over all the elements in the batch
                for i = 1, input:size()[1] do
                    self.gradInput[{{i}}] = input[i] - target -- grad should increase (actual - correct)
                end
                self.gradInput = self.gradInput * self.invariance_strength
            else
                self.gradInput:resizeAs(gradOutput)
                self.gradInput:fill(0)
            end
        else
            if self.force_invariance then
                self.gradInput:resizeAs(input)

                local target = torch.mean(input, 1) -- mean over all the elements in the batch
                for i = 1, input:size()[1] do
                    self.gradInput[{{i}}] = input[i] - target -- grad should increase (actual - correct)
                end
                self.gradInput = self.gradInput * self.invariance_strength

                self.gradInput[{{}, self.passthrough}] = gradOutput[{{}, self.passthrough}]
            else
                self.gradInput:resizeAs(gradOutput)
                self.gradInput:fill(0)
                self.gradInput[{{}, self.passthrough}] = gradOutput[{{}, self.passthrough}]
            end
        end
    else
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    end

    return self.gradInput
end
