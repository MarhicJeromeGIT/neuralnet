require 'sciruby'

module Helpers

# a random gaussian number distribution, mean 0, deviation 1            
@@generator = Distribution::Normal.rng(0,1)
def self.rng
  @@generator.call()
end

# return the labels
def self.read_labels(label_filename='train-labels-idx1-ubyte')
  labels = File.open(label_filename, 'rb') do |f|
    # MSB
    abort('not a mnist file') if f.read(4).unpack('N')[0] != 2049
    # Number of images/labels
    nbl = f.read(4).unpack('N')[0]
    #puts "reading labels..."
    labels = f.read(nbl).unpack('C*')
    #puts "#{labels.size} labels read!"
    labels
  end
end

# return the images data, number of pixel per row, num of pixel per column
# convert from [0,255] to [0,1] range
def self.read_images(image_filename='train-images-idx3-ubyte', max_img=nil)
  File.open(image_filename, 'rb') do |f|
    msb, nbi = f.read(8).unpack('N*')
    #0008     32 bit integer  28               number of rows 
    #0012     32 bit integer  28               number of columns 
    rows, cols = f.read(8).unpack('N*')
    abort('error rows,cols') unless rows == cols && cols == 28
    # The index in the last dimension (cols) change the fastest ie we read col by col, line by line
    nbi = [nbi, max_img].min if max_img
    data = f.read(nbi*rows*cols).unpack('C*')
    data = data.map{|c| c/255.0}.each_slice(rows*cols).to_a
    [data,rows,cols]
  end
end

def self.show_img(img,cols)
  i = 0
  s = ""
  img.each do |c|
    s += to_ascii(c)
    i += 1
    if i == cols
      i = 0
      puts s
      s = ""
    end
  end
  nil
end

def self.sigmoid z
  1.0 / (1.0 + Math.exp(-z))
end

def self.sigmoid_prime z
  sigmoid(z) * (1.0 - sigmoid(z) )
end

def self.z w,x,b
  abort("z error: #{w.count} != #{x.count}") if w.count != x.count
  (0...w.count).inject(0) {|r, i| r + w[i]*x[i] } + b
end

# w,x vectors, b a value
def self.sigmoid_neuron w,x,b
  # compute the dot product of w and x and add b
  abort("sigmoid neuron error: #{w.count} != #{x.count}") if w.count != x.count
  z = (0...w.count).inject(0) {|r, i| r + w[i]*x[i] } + b
  sigmoid z
end

# w,x vectors, b a value
def self.sigmoid_prime_neuron w,x,b
  # compute the dot product of w and x and add b
  abort("sigmoid prime neuron error: #{w.count} != #{x.count}") if w.count != x.count
  z = (0...w.count).inject(0) {|r, i| r + w[i]*x[i] } + b
  sigmoid_prime z
end


def self.answer_vec(answer, size=10)
  a = Array.new(10,0)
  a[answer] = 1
  a
end

def self.vector_length(v)
  sum2 = (0...v.count).inject(0){|r,i| r + v[i]**2 }
  Math.sqrt sum2
end

# return u - v
def self.vector_diff(u,v)
  abort "vector_diff: #{u.size} != #{v.size}" unless u.size == v.size
  (0...u.size).map{|i| u[i]-v[i] }
end

#Hadamard product
def self.vector_mul(u,v)
  abort "vector_mul: #{u.size} != #{v.size}" unless u.size == v.size
  (0...u.size).map{|i| u[i]*v[i] }
end

def self.vector_dot(u,v)
  abort "vector_dot: #{u.size} != #{v.size}" unless u.size == v.size
  (0...u.size).inject(0){|r,i| r + u[i]*v[i]}
end

private
  def self.to_ascii(c)
    outputs = [' ',"\u2591","\u2592","\u2592","\u2593","\u2593"]
    outputs[(c*5).floor]
  end
end


