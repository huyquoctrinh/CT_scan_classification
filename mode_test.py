class build_model(img_shape):
  def _init_(self, model1,model2,img_shape):
    self.model1 = model1(img_shape)
    self.model2 = model2(img_shape)
    self.img_shape = img_shape
  def model(self):
    inputs = tf.keras.Input(shape = self.img_shape)
    inputs = data_augmentation(inputs)
    inputs = preprocess_input(inputs)
    
    x_1 = self.model1(inputs,training =False)
    x_2 = self.model2(inputs,training =False)

    

def model2(IMG_SHAPE):
  base_model2 = tf.keras.applications.ResNet101(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
  return base_model2

  def model1(IMG_SHAPE):
  base_model1 = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
  return base_model1

  def conv_block(inputs, numFilters):
    
    # inputs = Input(input_shape =(128,128,3))
    
    tower_0 = Convolution2D(numFilters, (1,1), padding='same', kernel_initializer = 'he_normal')(inputs)
    tower_0 = BatchNormalization()(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    tower_1 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    tower_2 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_2)
    tower_2 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_3 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation("relu")(tower_3)
    inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    return inception_module