#!/usr/bin/env ruby

require 'sciruby'
require './helpers'
require 'pry'
require 'json'
require './neural_net'

label_file = 't10k-labels-idx1-ubyte' 
image_file = 't10k-images-idx3-ubyte'

net = NeuralNet.read_from_file 'net1'
net.test label_file, image_file, 100

