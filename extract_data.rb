# ruby extract_data.rb train-labels-idx1-ubyte train-images-idx3-ubyte

require 'sciruby'
require './helpers'
require 'pry'

# http://neuralnetworksanddeeplearning.com/chap1.html
class NeuralNet
  N = 15 # Neurons in hidden layer
  O = 10 # Neurons in output layer (because we classify between 10 digits)

  def initialize( label_filename='train-labels-idx1-ubyte', image_filename='train-images-idx3-ubyte' )
    @labels = Helpers.read_labels(label_filename)
    @images,@rows,@cols = Helpers.read_images(image_filename,5000)
    @images,@test_images = @images.each_slice(4000).to_a
    @size_input = @rows*@cols    
    
    # Initialize neurons with random weights and biases
    @weights_hidden = N.times.map do @size_input.times.map do Helpers.rng end end
    @biases_hidden  = N.times.map do Helpers.rng end

    @weights_output = O.times.map do N.times.map do Helpers.rng end end
    @biases_output = O.times.map do Helpers.rng end

  end

  def print_example
    5.times do |i|
      puts @labels[i]
      puts Helpers.show_img(@images[i],@cols)
    end
  end

  def to_s
    "#{@labels.size} image read. Size is #{@size_input} (#{@rows}x#{@cols})"
  end

  def step_for_image idx
    a_hidden = Array.new(N) #activation
    e_hidden = Array.new(N) #error
    abort "wrong weights sizes!" unless @weights_hidden.size == N && @biases_hidden.size == N
    (0...N).each do |i|
      z = Helpers.z @images[idx], @weights_hidden[i], @biases_hidden[i]
      a_hidden[i] = Helpers.sigmoid z
      e_hidden[i] = Helpers.sigmoid_prime z
    end

    #now the output layer:
    a_output = Array.new(O)
    e_output = Array.new(O)
    (0...O).each do |i|
      z = Helpers.z a_hidden, @weights_output[i], @biases_output[i]
      a_output[i] = Helpers.sigmoid z
      e_output[i] = Helpers.sigmoid_prime z
    end
 
    # Output error 
    # Compute the quadratic error:
    # y : the right answer (what we expect)
    answer_vec = Helpers.answer_vec( @labels[idx] )
    error_diff = Helpers.vector_diff(a_output,answer_vec)
    # This is BP1 from the book
    delta_output = Helpers.vector_mul(e_output, error_diff)

    # Now compute the error on the previous (hidden) layer
    # BP2 in the book
    delta_hidden = @weights_output.transpose.map do |v|
       abort "delta_hidden #{v.size} != #{delta_output.size}" unless v.size == delta_output.size
       Helpers.vector_dot v, delta_output
    end
    delta_hidden = Helpers.vector_mul( delta_hidden, e_hidden)

    # Gradient descente:
    # output layer:
    # see BP4 in the book
    lambda = 1.0
    @weights_output = (0...O).map do |i|
      v = @weights_output[i] # the weights for the ith neurone
      di = delta_output[i]
      delta = a_hidden.map{|val| val * di * lambda }
      Helpers.vector_diff( v, delta)
    end

    @biases_output = (0...O).map do |i|
      v = @biases_output[i] # the biases for the ith neurone   
      delta = delta_output[i]
      v - lambda * delta
    end
 
    # hidden layer
    @weights_hidden = (0...N).map do |i|
      v = @weights_hidden[i] # the weights for the ith neurone
      di = delta_hidden[i]
      delta = @images[idx].map{|val| val * di * lambda }
      Helpers.vector_diff( v, delta)
    end

    @biases_output = (0...N).map do |i|
      v = @biases_hidden[i] # the biases for the ith neurone   
      delta = delta_hidden[i]
      v - lambda * delta
    end


    a_output
   
  end

  def test
    test_size = 1000
    correct_answers = 0
    (0...@images.size).to_a.sample(test_size).each do |i|
      r = step_for_image i
      Helpers.show_img @images[i],@cols
      puts "Image is a #{r.each_with_index.sort.to_a[-1][1]} (or a  #{r.each_with_index.sort.to_a[-2][1]}) (real answer: #{@labels[i]})"
      #answer_vec = Helpers.answer_vec( @labels[i] )
      correct_answers += 1 if r.each_with_index.max[1] == @labels[i]
    end
    puts "Correct answer rate : #{correct_answers*100/test_size}%"
  end

  def learn
    batch_size = 4000
    (0...@images.size).to_a.sample(batch_size).each do |i|
      # the answer i get
      r = step_for_image i
      # the real answer
      #answer_vec = Helpers.answer_vec( @labels[i] )

      #delta_w = N.times.map do @size_input.times.map do 0 end end
      #delta_b = O.times.map do N.times.map do 0 end end
    end
  end
end

label_file, image_file = ARGV
net = NeuralNet.new(label_file, image_file)
net.learn()
net.test()











  









