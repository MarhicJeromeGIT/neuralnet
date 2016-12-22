require 'sciruby'
require './helpers'
require 'pry'
require 'json'

# http://neuralnetworksanddeeplearning.com/chap1.html
class NeuralNet
  LEARNING_ITERATIONS_DEFAULT = 120
  MINI_BATCH_SAMPLES_DEFAULT  = 500
  LAMBDA_DEFAULT = 1.0 #the learning rate

  attr_accessor :activations
  attr_accessor :output_error

  def initialize args={}
    @lambda              = args['lambda'] || LAMBDA_DEFAULT
    @learning_iterations = args['learning_iterations'] || LEARNING_ITERATIONS_DEFAULT
    @mini_batch_size     = args['mini_batch_size'] || MINI_BATCH_SAMPLES_DEFAULT   
  end

  class Layer
    @@layers = 0
    attr_accessor :activation
    attr_accessor :error
    attr_accessor :weights 
    attr_accessor :biases
    attr_accessor :input_size
    attr_accessor :output_size
   
    def initialize args
      @layer_index = @@layers
      @@layers += 1
      puts "initializing layer #{@layer_index}: #{args['input_size']} -> #{args['output_size']}"
      @output_size = args['output_size'] # number of neurons
      @input_size  = args['input_size']
      @weights = args['weights'] || @output_size.times.map do @input_size.times.map do Helpers.rng end end
      abort "wrong weights size" unless @weights.size == @output_size && @weights[0].size == @input_size
      @biases  = args['biases'] || @output_size.times.map do Helpers.rng end 
      abort "wrong bias size" unless @biases.size == @output_size 
    end
    
    def to_json arg
      JSON.generate( { 
        :input_size => @input_size, 
        :output_size => @output_size, 
        :weights => @weights, 
        :biases => @biases 
      }) 
    end
   
    def self.from_json layer
      Layer.new layer    
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
    def SGD(input, lambda)
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

  def save_to_file filename
    File.open(filename, 'w') do |f|
      f.write self.to_json
    end
  end

  def self.read_from_file filename
    net = NeuralNet.new
    File.open(filename, 'r') do |f|
      data = JSON.parse f.read
      net.from_json data
    end
    return net
  end

  def from_json data
    @layers = Array.new
    data.each do |l|
      @layers << Layer.from_json(l)
    end
  end

  def to_json 
   JSON.generate(@layers)   
  end

  def step_for_image input, answer_vec
    @activations = Array.new
    @activations << input 
    @layers.each do |layer|
      @activations << (layer.forward @activations.last)
    end
    
    # Output error
    # Compute the quadratic error:
    # y : the right answer (what we expect)
    puts "answer: #{answer_vec.join(',')}" 
    @output_error = Helpers.vector_diff(@activations.last,answer_vec)
    delta_error = @output_error
    # We start from the last layer
    @layers.reverse.each do |layer|
      delta_error = layer.backward delta_error
      delta_error = layer.weights.transpose.map do |v|
        abort "delta_hidden #{v.size} != #{delta_error.size}" unless v.size == delta_error.size
        Helpers.vector_dot v, delta_error
      end
    end

    # Gradient descente:
    # output layer:
    # see BP4 in the book
    (0...@layers.size).each do |i|
      @layers[i].SGD @activations[i], @lambda
    end
  end

  def test label_filename, image_filename, max_img=nil
    @labels = Helpers.read_labels(label_filename)
    @images,@rows,@cols = Helpers.read_images(image_filename, max_img)
    @size_input = @rows*@cols

    test_size = @images.size
    correct_answers = 0
    (0...@images.size).each do |i|
      @activation_hidden = @layers[0].forward @images[i]
      @activation_output = @layers[1].forward @activation_hidden
      sorted_results = @activation_output.each_with_index.to_a.sort
      if sorted_results[-1][1] == @labels[i]
        correct_answers += 1
      else
        Helpers.show_img @images[i],@cols
        puts "Image is a #{sorted_results[-1][1]} (or a  #{sorted_results[-2][1]}) (real answer: #{@labels[i]})"
      end
    end
    puts "Correct answer rate : #{correct_answers*100/test_size}%"
  end

  def learn args
    @inputs = args['inputs']
    @answers = args['answers']
    abort "must provide input and answers" unless @inputs && @answers
    abort "input and answers are not the same size" unless @inputs.count == @answers.count

    layers_size = args['layers_size']
    weights = args['weights']
    biases = args['biases'] 
    @layers = Array.new()
    (0...layers_size.count-1).each do |i|
      @layers << (Layer.new({
                    'input_size'  => layers_size[i], 
                    'output_size' => layers_size[i+1],
                    'weights'     => weights&.at(i),
                    'biases'      => biases&.at(i)
                 }))
    end

    # Some verifications :
    abort "layer size must be >= 2" unless layers_size.count >= 2
    abort "wrong first layer size" unless @inputs[0].count == @layers[0].input_size
    abort "wrong last layer size" unless @answers[0].count == @layers[-1].output_size

    (1..@learning_iterations).each do |i|
       (0...@inputs.count).to_a.sample(@mini_batch_size).each do |i|
         step_for_image @inputs[i], @answers[i]
         yield i
      end
    end
  end
end


