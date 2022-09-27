from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Conv3D,
    AveragePooling3D,
    MaxPooling3D,
    add,
    BatchNormalization,
    Conv3DTranspose,
    concatenate,
    UpSampling3D,
    Multiply,
    Lambda
)

def dmn_model(input_shape, block_fn=basic_block, reg_factor=1e-4):

  _handle_data_format()
  if len(input_shape) != 4:
      raise ValueError("Input shape should be a tuple "
                        "(conv_dim1, conv_dim2, conv_dim3, channels) "
                        "for tensorflow as backend or "
                        "(channels, conv_dim1, conv_dim2, conv_dim3) "
                        "for theano as backend")

  block_fn = _get_block(block_fn)
  input = Input(shape=input_shape)
  
  # first conv
  conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                          strides=(2, 2, 2),
                          kernel_regularizer=l2(reg_factor)
                          )(input)
  # max pooling
  pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1),
                      padding="same")(conv1)

  block = pool1
  filters = 64
  list_of_blocks = []
  for i, r in enumerate([2, 2, 2, 2]):
    block = _residual_block3d(block_fn, filters=filters,
                              kernel_regularizer=l2(reg_factor),
                              repetitions=r, is_first_layer=(i == 0)
                              )(block)
    list_of_blocks.append(block)
    filters *= 2

  list_of_feature_blocks = []
  resize = 2
  for i, block in enumerate(list_of_blocks):
    feature_block = _bn_relu_conv3d(filters=128,
                  kernel_size=(1, 1, 1),
                  padding="valid",
                  strides=(1, 1, 1),
                  kernel_regularizer=l2(reg_factor)
                  )(block)
    if i == 0:
      feature_block = AveragePooling3D((2, 2, 2))(feature_block)
    elif i > 1:
      #feature_block = Conv3DTranspose(128, (resize, resize, resize), strides=(resize, resize, resize), padding='same')(feature_block)
      feature_block = UpSampling3D(resize)(feature_block)
      resize *= 2

    list_of_feature_blocks.append(feature_block)

  multiscale_feature = concatenate([list_of_feature_blocks[0], list_of_feature_blocks[1], list_of_feature_blocks[2], list_of_feature_blocks[3]], axis=-1)

  filters = 512
  conv = multiscale_feature
  sizes = [(3, 3, 3), (1, 1, 1), (3, 3, 3), (1, 1, 1)]
  for i, kernel in enumerate(sizes):
    conv = _conv_bn_relu3D(filters=filters, kernel_size=kernel,
                            padding="same",
                            strides=(1, 1, 1),
                            kernel_regularizer=l2(reg_factor)
                            )(conv)
    filters //= 2

  conv = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv)
  output = Conv3DTranspose(1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv)
  att_map_output = Activation("sigmoid", name='att_map_output')(output)

  model = Model(inputs=input, outputs=att_map_output)
  return model