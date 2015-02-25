require 'nn'

local SelectiveOutputClamp, parent = torch.class('nn.SelectiveOutputClamp', 'nn.Module')

--[[

This layer takes a batch and sets all samples in the batch equal to the first one.
A single index of each sample, `self.passthrough`, can be sent on unchanged (not clamped).

That is,

self.output[i] = input[1] -- all outputs in batch same as first input
self.output[{{}, self.passthrough}] = input[{{}, self.passthrough}]
                    -- one value of each sample in input sent through unchanged

Gradients are passed through unchanged.

First sample in a batch (unchanged):
-> | ->
-> | ->
-> | ->
-> | ->
-> | ->
->   ->     -- this is the passthrough
-> | ->
-> | ->

Other samples of the batch:
(all clamped to input[1] except for the passthrough)
~> | ->
~> | ->
~> | ->
~> | ->
~> | ->
~>   ~>     -- this is the passthrough (unchanged)
~> | ->
~> | ->

Gradients (unchanged):
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
    -- parent.__init(self)
    self:reset()
end

-- Set which ouput index will pass through unimpeded.
-- Warning: treats the first index of its input as a batch index.
function SelectiveOutputClamp:setPassthroughIndex(index)
    self.passthrough = index
end

function SelectiveOutputClamp:reset()
    self.passthrough = nil
    self.active = true
end

function SelectiveOutputClamp:updateOutput(input)
    if self.active then
        self.output = input:clone()
        for i = 1, input:size()[1] do
            self.output[i] = input[1]:clone()
        end

        if self.passthrough ~= nil then
           self.output[{{}, self.passthrough}] = input[{{}, self.passthrough}]
        end
    else
        self.output = input:clone()
    end

    return self.output
end

function SelectiveOutputClamp:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

















