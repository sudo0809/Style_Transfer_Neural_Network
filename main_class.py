import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from matplotlib import pyplot as plt

class Style_image:

  def __init__(self,content_img_path, style_img_path, max_dim=256, iterations=500,
          style_weight=1, content_weight=250, tv_weight=10):

    self.content_img_path = content_img_path
    self.style_img_path = style_img_path
    self.max_dim = max_dim
    self.iterations = iterations
    #weights
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.total_variation_weight = tv_weight
    #layer names
    self.style_layer_names = ["block1_conv2",
                              "block2_conv1",
                              "block3_conv1",
                              "block4_conv1",
                              "block5_conv1",]

    self.content_layer_name = "block5_conv2"
    #Starting look of generation Image
    self.gen_image_type = "content_image"

  def set_content_layers(self, content_layer_name):
    self.content_layer_name = content_layer_name

  def set_style_layers(self, style_layer_name_list):
    self.style_layer_names = style_layer_name_list

  def set_noise_gen_image(self):
    print("Generation Image set to Noise Image")
    self.gen_image_type = "noise_image"

  def set_gen_image(self, gen_image_path):
    self.gen_image_type = "new_image"


  def preprocessing_img(self, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(self.img_nrows, self.img_ncols))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

  def deprocess_img(self, x):
    x = x.reshape((self.img_nrows,self.img_ncols,3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.77
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

  def gram_matrix(self,x):
    x = tf.transpose(x, (2,0,1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

  def content_loss(self,base, combination):
    return tf.reduce_sum(tf.square(combination - base))

  def style_loss(self, style, combination):
    # img_nrows, img_ncols = target_shape
    s = self.gram_matrix(style)
    c = self.gram_matrix(combination)
    channels = 3
    size = self.img_nrows * self.img_ncols
    return tf.reduce_sum(tf.square(s-c))/(4.0 *(channels**2)*(size**2))

  def total_variation_loss(self, x):
    # img_nrows, img_ncols = target_shape
    a = tf.square(x[:, :self.img_nrows-1, :self.img_ncols-1, :] - x[:, 1:, :self.img_ncols-1, :])
    b = tf.square(x[:, :self.img_nrows-1, :self.img_ncols-1, :] - x[:, :self.img_nrows-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

  def get_feature_extractor(self):
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    output_dict = dict([(layer.name, layer.output) for layer in model.layers])
    return keras.Model(inputs=model.inputs, outputs=output_dict)

  def compute_loss(self, combination_img, content_img, style_img):
	    
    input_tensor = tf.concat([content_img, style_img, combination_img], axis=0)
    features_extractor = self.get_feature_extractor()
    features = features_extractor(input_tensor)

    #initialize the loss
    loss = tf.zeros(shape=())

    # add content loss
    layer_features = features[self.content_layer_name]
    content_img_feature = layer_features[0, :, :, :]
    combination_img_feature = layer_features[2, :, :, :]
    loss += self.content_weight * self.content_loss(content_img_feature, combination_img_feature)

    #Add style loss
    for layer_name in self.style_layer_names:
      layer_features = features[layer_name]
      style_img_features = layer_features[1, :, :, :]
      combination_img_features = layer_features[2, :, :, :]
      sl = self.style_loss(style_img_features, combination_img_features)
      loss += sl * (self.style_weight/len(self.style_layer_names))

    # Add total variation loss
    loss += self.total_variation_weight * self.total_variation_loss(combination_img)

    return loss

  def compute_loss_and_grads(self, combination_img, content_img, style_img):
    with tf.GradientTape() as tape:
      loss = self.compute_loss(combination_img, content_img, style_img)
    grads = tape.gradient(loss, combination_img)
    return loss, grads

  def train(self):
    width, height = keras.preprocessing.image.load_img(self.content_img_path).size
    self.img_nrows = self.max_dim
    self.img_ncols = int(width * self.img_nrows / height)
    target_shape = (self.img_nrows, self.img_ncols)

    content_image = self.preprocessing_img(self.content_img_path)
    style_image = self.preprocessing_img(self.style_img_path)

    #generation Image used
    if self.gen_image_type == "noise_image":
      img = np.random.rand(self.img_nrows, self.img_ncols, 3) * 255
      # img.astype("float32")
      img = np.expand_dims(img, axis=0).astype("float32")
      img = tf.keras.applications.vgg19.preprocess_input(img)
      img = tf.convert_to_tensor(img)
      combination_image = tf.Variable(img)
    elif self.gen_image_type == "new_image":
      combination_image = tf.Variable(preprocessing_img(self.gen_image_type))
    else:
      combination_image = tf.Variable(content_image)


    optimizer = keras.optimizers.Adam(
    keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))

		# self.iterations = 400
    for i in range(1, self.iterations+1):
      loss, grads = self.compute_loss_and_grads(combination_image, content_image, style_image)
      optimizer.apply_gradients([(grads, combination_image)])
      if i % 50 == 0:
        print("Iteration %d: loss=%.2f" %(i, loss))
        img = self.deprocess_img(combination_image.numpy())
        fname = "gen_image"+"_at_iteration_%d.png" %i
        plt.imshow(img)
        plt.show()


