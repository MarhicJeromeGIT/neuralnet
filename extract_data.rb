# ruby extract_data.rb train-labels-idx1-ubyte train-images-idx3-ubyte

require 'sciruby'
require './helpers'
require 'pry'

# http://neuralnetworksanddeeplearning.com/chap1.html
class NeuralNet
  N = 15 # Neurons in hidden layer
  O = 10 # Neurons in output layer (because we classify between 10 digits)

  class Layer
    @@layers = 0
    attr_accessor :activation
    attr_accessor :error
    attr_accessor :weights 
    attr_accessor :biases
   
    def initialize(input_size,output_size)
      @layer_index = @@layers
      @@layers += 1
      puts "initializing layer : #{input_size} -> #{output_size}"
      @output_size = output_size # number of neurons
      @input_size = input_size
      @weights = @output_size.times.map do @input_size.times.map do Helpers.rng end end
      @biases = @output_size.times.map do Helpers.rng end  
    end
    
    def forward(input)
      abort "forward: inputs size #{input.size} is not #{@input_size}" unless input.size == @input_size
      #puts "forward : #{@layer_index} #{@input_size} #{@weights.size} #{@biases.size}"
      
      @activation = Array.new(@output_size)
      @error = Array.new(@output_size)
      (0...@output_size).each do |i|
        z = Helpers.z input, @weights[i], @biases[i]
        @activation[i] = Helpers.sigmoid z
        @error[i] = Helpers.sigmoid_prime z
      end
      @activation
    end
    
    def backward(delta)
      abort "backward: delta size #{delta.size} is not #{output_size}" unless delta.size == @output_size
      @delta = Helpers.vector_mul( delta, @error)
      abort "backward: delta is nil" unless @delta
      @delta
    end

    # Gradient descente:
    def SGD(input, lambda = 1.0)
      abort "SGD: inputs size #{input.size} is not #{@input_size}" unless input.size == @input_size
      abort "SGD #{@layer_index}: delta is nil" unless @delta

      # see BP4 in the book
      @weights = (0...@output_size).map do |i|
        v = @weights[i] # the weights for the ith neurone
        di = @delta[i]
        delta = input.map{|val| val * di * lambda }
        Helpers.vector_diff( v, delta)
      end

      @biases = (0...@output_size).map do |i|
        v = @biases[i] # the biases for the ith neurone   
        di = @delta[i]
        v - lambda * di
      end
    end  
  end

  def initialize( label_filename='train-labels-idx1-ubyte', image_filename='train-images-idx3-ubyte' )
    @labels = Helpers.read_labels(label_filename)
    @images,@rows,@cols = Helpers.read_images(image_filename,60000)
    @images,@test_images = @images.each_slice(50000).to_a
    @size_input = @rows*@cols    

    @layers = Array.new(2)
    @layers[0] = Layer.new(@rows*@cols, N)
    @layers[1] = Layer.new(N,10)
  end

  def step_for_image idx
    @activation_hidden = @layers[0].forward @images[idx]
    @activation_output = @layers[1].forward @activation_hidden
    
    # Output error
    # Compute the quadratic error:
    # y : the right answer (what we expect)
    answer_vec = Helpers.answer_vec( @labels[idx] )
    error_diff = Helpers.vector_diff(@activation_output,answer_vec)
    # This is BP1 from the book
    @delta = @layers[1].backward error_diff

    # Now compute the error on the previous (hidden) layer
    # BP2 in the book
    delta_hidden = @layers[1].weights.transpose.map do |v|
       abort "delta_hidden #{v.size} != #{@delta.size}" unless v.size == @delta.size
       Helpers.vector_dot v, @delta
    end
    @layers[0].backward delta_hidden

    # Gradient descente:
    # output layer:
    # see BP4 in the book
    @layers[0].SGD @images[idx]
    @layers[1].SGD @activation_hidden
  end

  def test
    test_size = 5000
    correct_answers = 0
    (0...@images.size).to_a.sample(test_size).each do |i|
      @activation_hidden = @layers[0].forward @images[i]
      @activation_output = @layers[1].forward @activation_hidden
      sorted_results = @activation_output.each_with_index.to_a.sort
      Helpers.show_img @images[i],@cols
      puts "Image is a #{sorted_results[-1][1]} (or a  #{sorted_results[-2][1]}) (real answer: #{@labels[i]})"
      correct_answers += 1 if sorted_results[-1][1] == @labels[i]
    end
    puts "Correct answer rate : #{correct_answers*100/test_size}%"
  end

  def learn
    batch_size = 40000
    (0...@images.size).to_a.sample(batch_size).each do |i|
      r = step_for_image i
    end
  end
end

label_file, image_file = ARGV
net = NeuralNet.new(label_file, image_file)
net.learn()
net.test()











  









