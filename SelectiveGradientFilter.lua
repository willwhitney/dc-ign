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

function SelectiveGradientFilter:setPassthroughIndex(index)
    self.passthrough = index
end

function SelectiveGradientFilter:reset()
    self.passthrough = nil
    self.active = true
end

function SelectiveGradientFilter:updateOutput(input)
    self.output = input
    return self.output
end

function SelectiveGradientFilter:updateGradInput(input, gradOutput)
    if self.active then
        if self.passthrough == nil then
            self.gradInput:resizeAs(gradOutput)
            self.gradInput:fill(0)
            self.gradInput[1] = gradOutput[1] -- let it learn from one sample per batch
        else
            self.gradInput:resizeAs(gradOutput)
            self.gradInput:fill(0)
            self.gradInput[{{}, self.passthrough}] = gradOutput[{{}, self.passthrough}]
            self.gradInput[1] = gradOutput[1] -- let it learn from one sample per batch
        end
    else
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    end

    return self.gradInput
end
