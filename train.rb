#!/usr/bin/env ruby

require 'sciruby'
require './helpers'
require 'pry'
require 'json'
require './neural_net'

label_filename = 'train-labels-idx1-ubyte'
image_filename = 'train-images-idx3-ubyte'
max_img = 1
# Read the training set 
#labels = Helpers.read_labels( label_filename )
#inputs,rows,cols = Helpers.read_images( image_filename, max_img )
#size_input = rows*cols
#puts "input size: #{size_input}"
# Format the labels into vectors :
#answers = labels[0...max_img].map{|l| Helpers.answer_vec(l) }
#size_output = 10

size_input = 1
inputs = [[1]]
answers = [[0]]
size_output = 1
weights = [[[0.6]]]
biases = [[0.9]]

puts "start training..."
net = NeuralNet.new({ 
  'learning_iterations' => 300, 
  'mini_batch_size' => 1,
  'lambda' => 0.15 
})

net.learn({
  'inputs'      => inputs,
  'answers'     => answers,
  'layers_size' => [size_input,size_output],
  'weights'     => weights,
  'biases'      => biases
}) do |i|
  # This block is called after each iteration
  puts "output : #{net.activations.last.map{|v| v.round(2)}.join(',')}"
  puts "Error : #{(Helpers.vector_length net.output_error).round(2)}"
end

net.save_to_file 'net2'
puts "training finished and weights written to file net2"

