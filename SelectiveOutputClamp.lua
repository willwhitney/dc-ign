require 'nn'

local SelectiveOutputClamp, parent = torch.class('nn.SelectiveOutputClamp', 'nn.Module')

--[[

The API here is a little tricky, so be careful.

When it's created, the SelectiveOutputClamp has nil output and nil passthrough.

Call `setPassthroughIndex` with the index of the data you want it to pass
through unchanged.

The first time it runs updateOutput, it will save the input it receives as its output.
These are its clamped values.

From then until it is reset, its output for all indices other than `self.passthrough`
will be fixed as those clamped values.

Gradients are passed through unchanged.

First run:
-> | ->
-> | ->
-> | ->
-> | ->
-> | ->
->   ->     -- this is the passthrough
-> | ->
-> | ->

Additional runs:
~> | ->
~> | ->
~> | ->
~> | ->
~> | ->
~>   ~>
~> | ->
~> | ->

Gradients:
<- | <-
<- | <-
<- | <-
<- | <-
<- | <-
<-   <-
<- | <-
<- | <-

--]]


function SelectiveOutputClamp:__init()
    parent.__init(self)
    self.reset()
end

-- Set which ouput index will pass through unimpeded.
-- Warning: treats the first index of its input as a batch index.
function SelectiveOutputClamp:setPassthroughIndex(index)
    self.passthrough = index
end

function SelectiveOutputClamp:reset()
    self.output = nil
    self.passthrough = nil
end

function SelectiveOutputClamp:updateOutput(input)
    if self.passthrough == nil or self.output == nil then
        self.output = input
    else
        self.output[{{}, self.passthrough}] = input[{{}, self.passthrough}]
    end

    return self.output
end

function SelectiveOutputClamp:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end
