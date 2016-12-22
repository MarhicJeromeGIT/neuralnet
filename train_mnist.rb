#!/usr/bin/env ruby

require 'sciruby'
require './helpers'
require 'pry'
require 'json'
require './neural_net'
require 'nyaplot'

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

learning_iterations = 300
puts "start training..."
net = NeuralNet.new({ 
  'learning_iterations' => learning_iterations, 
  'mini_batch_size' => 1,
  'lambda' => 0.15 
})

plot = Nyaplot::Plot.new

def learn_and_plot net, params
  error_costs = Array.new
  net.learn(params) do |i|
    # This block is called after each iteration
    #puts "output : #{net.activations.last.map{|v| v.round(2)}.join(',')}"
    #puts "Error : #{(Helpers.vector_length net.output_error).round(2)}"
    error_cost = (Helpers.vector_length net.output_error) 
    puts error_cost
    error_costs << error_cost
  end
  error_costs
end

params = {
  'inputs'      => [[1]],
  'answers'     => [[0]],
  'layers_size' => [1,1],
  'weights'     => [[[0.6]]],
  'biases'      => [[0.9]]
}

error_cost = learn_and_plot net, params
plot.add(:scatter, (1..learning_iterations).to_a, error_cost)

# Change some params:
params['weights'] = [[[2]]]
params['biases'] = [[2]]
error_cost = learn_and_plot net, params
plot.add(:scatter, (1..learning_iterations).to_a, error_cost)

plot.export_html('plot.html')



net.save_to_file 'net2'
puts "training finished and weights written to file net2"

