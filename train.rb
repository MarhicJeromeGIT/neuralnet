require 'sciruby'
require './helpers'
require 'pry'
require 'json'
require './neural_net'

label_file = 'train-labels-idx1-ubyte'
image_file = 'train-images-idx3-ubyte'

puts "start training..."
net = NeuralNet.new({ 'learning_iterations' => 10, 'mini_batch_size' => 20 })
net.learn label_file, image_file, 1000
net.save_to_file 'net1'
puts "training finished and weights written to file net1"

