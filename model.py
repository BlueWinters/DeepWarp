
import tensorflow as tf
import tensorflow.contrib.slim as slim



class DeepGaze:
    def __init__(self, batch_size, name):
        self.height = 41*1
        self.width = 51*1
        self.angle_dim = 2
        self.lm_pos_dim = 7*2
        self.light_on = False
        self.batch_size = batch_size
        self.name = name

    def conv2d_batch_norm_relu(self, input, out_chl, kernel, stride, padding, name):
        # basic implement
        with tf.variable_scope(name):
            body = tf.layers.conv2d(input, out_chl, kernel, stride, padding, use_bias=False, name='conv2d')
            body = tf.layers.batch_normalization(body, momentum=0.9, training=self.is_train, name='bn')
            body = tf.nn.relu(body, name='relu')
            return body

    def meshgrid(self, height, width, ones_flag=None):
        # get the mesh-grid in a special area(-1,1)
        # output:
        #   @shape --> 2,H*W
        #   @explanation --> (0,:) means all x-coordinate in a mesh
        #                    (1,:) means all y-coordinate in a mesh
        with tf.variable_scope('meshgrid'):
            y_linspace = tf.linspace(-1., 1., height)
            x_linspace = tf.linspace(-1., 1., width)
            x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
            x_coordinates = tf.reshape(x_coordinates, shape=[-1])
            y_coordinates = tf.reshape(y_coordinates, shape=[-1])
            if ones_flag is None:
                indices_grid = tf.stack([x_coordinates, y_coordinates], axis=0)
            else:
                indices_grid = tf.stack([x_coordinates, y_coordinates, tf.ones_like(x_coordinates)], axis=0)
            return indices_grid

    def repeat(self, x, n_repeats):
        with tf.variable_scope('_repeat'):
            # rep = tf.transpose(
            #     tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            # rep = tf.cast(rep, 'int32')
            rep = tf.reshape(tf.ones(shape=tf.stack([n_repeats, ]), dtype=tf.int32), (1, n_repeats))
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def interpolate(self, input, x, y, out_height, out_width, name):
        with tf.variable_scope(name):
            N, H, W, C = input.get_shape().as_list()

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
            H_f = tf.cast(H, dtype=tf.float32)
            W_f = tf.cast(W, dtype=tf.float32)

            # scale indices from [-1,1] --> [0,W] or [0,H]
            x = (x + 1.0) * (W_f - 1) * 0.5 # ? W_f-1, different, 5.11 modify
            y = (y + 1.0) * (H_f - 1) * 0.5
            # get x0 and x1 in bilinear interpolation
            x0 = tf.cast(tf.floor(x), tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), tf.int32)
            y1 = y0 + 1

            # clip the coordinate value
            max_y = tf.cast(H - 1, dtype=tf.int32)
            max_x = tf.cast(W - 1, dtype=tf.int32)
            zero = tf.constant([0], shape=(1,), dtype=tf.int32)
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            # some bad implement, maybe there is another good one
            # ref: https://github.com/kevinzakka/spatial-transformer-network/blob/master/transformer.py#L158
            flat_image_dimensions = H * W
            pixels_batch = tf.range(N) * flat_image_dimensions
            flat_output_dimensions = out_height * out_width
            base = self.repeat(pixels_batch, flat_output_dimensions)
            base_y0 = base + y0 * W
            base_y1 = base + y1 * W
            indices_a = base_y0 + x0
            indices_b = base_y1 + x0
            indices_c = base_y0 + x1
            indices_d = base_y1 + x1

            # gather every pixel value
            flat_image = tf.reshape(input, shape=(-1, C))
            flat_image = tf.cast(flat_image, dtype=tf.float32)
            pixel_values_a = tf.gather(flat_image, indices_a)
            pixel_values_b = tf.gather(flat_image, indices_b)
            pixel_values_c = tf.gather(flat_image, indices_c)
            pixel_values_d = tf.gather(flat_image, indices_d)

            x0 = tf.cast(x0, tf.float32)
            x1 = tf.cast(x1, tf.float32)
            y0 = tf.cast(y0, tf.float32)
            y1 = tf.cast(y1, tf.float32)

            area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
            area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
            area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
            area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
            output = tf.add_n([area_a * pixel_values_a,
                               area_b * pixel_values_b,
                               area_c * pixel_values_c,
                               area_d * pixel_values_d])
            # for mask the interpolate part which pixel don't move
            mask = area_a + area_b + area_c + area_d
            output = (1 - mask) * flat_image + mask * output
            return output

    def get_pixel_value_by_index(self, input, x, y, out_height, out_width, name):
        with tf.variable_scope(name):
            N, H, W, C = input.get_shape().as_list()

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
            H_f = tf.cast(H, dtype=tf.float32)
            W_f = tf.cast(W, dtype=tf.float32)

            # scale indices from [-1,1] --> [0,W] or [0,H]
            x = (x + 1.0) * (W_f - 1) * 0.5 # ? W_f-1, different, 5.11 modify
            y = (y + 1.0) * (H_f - 1) * 0.5
            # get x0 and x1 in bilinear interpolation
            x0 = tf.cast(tf.floor(x), tf.int32)
            y0 = tf.cast(tf.floor(y), tf.int32)

            # clip the coordinate value
            max_y = tf.cast(H - 1, dtype=tf.int32)
            max_x = tf.cast(W - 1, dtype=tf.int32)
            zero = tf.constant([0], shape=(1,), dtype=tf.int32)
            x = tf.clip_by_value(x0, zero, max_x)
            y = tf.clip_by_value(y0, zero, max_y)

            # some bad implement, maybe there is another good one
            # ref: https://github.com/kevinzakka/spatial-transformer-network/blob/master/transformer.py#L158
            flat_image_dimensions = H * W
            pixels_batch = tf.range(N) * flat_image_dimensions
            flat_output_dimensions = out_height * out_width
            base = self.repeat(pixels_batch, flat_output_dimensions)
            indices = base + y * W + x

            # gather every pixel value
            flat_image = tf.reshape(input, shape=(-1, C))
            flat_image = tf.cast(flat_image, dtype=tf.float32)
            pixel_values = tf.gather(flat_image, indices)
            return pixel_values

    def stn_sample(self, input, theta, name):
        with tf.variable_scope(name):
            N, iH, iW, iC = input.get_shape().as_list()
            _, fH, fW, fC = theta.get_shape().as_list()
            assert iH == fH and iW == fW and iC == 3 and fC == 3
            # re-order & reshape: N,H,W,C --> N,C,H*W
            theta = tf.reshape(theta, [-1, fC * fH * fW])
            theta = tf.layers.dense(theta, 6, activation=tf.nn.relu, name='dense')
            theta = tf.reshape(theta, (N, 2, 3))

            # get mesh-grid, 2,H*W
            indices_grid = self.meshgrid(self.height, self.width, ones_flag=True)
            indices_grid = tf.tile(tf.expand_dims(indices_grid, axis=0), (N, 1, 1))
            # affine matrix
            transformed_grid = tf.matmul(theta, indices_grid)

            # N,2,H*W + 2,H*W --> N,2,H*W
            # it represents the pixel-flow
            # 1.flow come from coarse_flow/fine_flow, and they limited in [-1,1](as as result tanh activation)
            # 2.indices_grid come from the line-space between [-1,1]
            # -------------------------------------------------------------------------------------------
            # NOTICE: this is a flow operation, and some of the value in result will overflow the original [-1,1],
            #         so a clip operation will implement in the interpolate function later.
            # -------------------------------------------------------------------------------------------
            # transformed_grid = tf.add(theta, indices_grid)

            # just like what it mean in STN, x_s and y_s respectively represents pixel value interpolation coordinate in
            # the original input image
            x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_image = self.interpolate(input, x_s_flatten, y_s_flatten, iH, iW, 'interpolate')
            # print(transformed_image.get_shape().as_list())
            transformed_image = tf.reshape(transformed_image, [N, iH, iW, iC])
            return transformed_image

    def bilinear_sample(self, input, flow, name):
        # reference to spatial transform network
        # 1.details can be found in office release:
        #   https://github.com/tensorflow/models/blob/master/research/transformer/spatial_transformer.py
        # 2.maybe another good implement can be found in:
        #   https://github.com/kevinzakka/spatial-transformer-network/blob/master/transformer.py
        #   but this one maybe contain some problems, go to --> https://github.com/kevinzakka/spatial-transformer-network/issues/10
        with tf.variable_scope(name):
            N, iH, iW, iC = input.get_shape().as_list()
            _, fH, fW, fC = flow.get_shape().as_list()
            assert iH == fH and iW == fW and iC == 3 and fC == 2
            # re-order & reshape: N,H,W,C --> N,C,H*W
            flow = tf.reshape(tf.transpose(flow, [0, 3, 1, 2]), [-1, fC, fH * fW])

            # get mesh-grid, 2,H*W
            indices_grid = self.meshgrid(self.height, self.width)

            # N,2,H*W + 2,H*W --> N,2,H*W
            # it represents the pixel-flow
            # 1.flow come from coarse_flow/fine_flow, and they limited in [-1,1](as as result tanh activation)
            # 2.indices_grid come from the line-space between [-1,1]
            # -------------------------------------------------------------------------------------------
            # NOTICE: this is a flow operation, and some of the value in result will overflow the original [-1,1],
            #         so a clip operation will implement in the interpolate function later.
            # -------------------------------------------------------------------------------------------
            transformed_grid = tf.add(flow, indices_grid)

            # just like what it mean in STN, x_s and y_s respectively represents pixel value interpolation coordinate in
            # the original input image
            x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_image = self.interpolate(input, x_s_flatten, y_s_flatten, iH, iW, 'interpolate')
            # print(transformed_image.get_shape().as_list())
            transformed_image = tf.reshape(transformed_image, [N, iH, iW, iC])
            return transformed_image

    def spatial_softmax_across_channels(self, weight):
        # do softmax activation across different channels
        weight = tf.cast(weight, dtype=tf.float32)
        N, H, W, C = weight.get_shape().as_list()
        weight = tf.reshape(weight, [-1, C])
        return tf.reshape(tf.nn.softmax(weight), [-1, H, W, C])

    def pixel_light_weight(self, input, light_weight):
        # perform softmax on light-weight across channels
        light_weight = self.spatial_softmax_across_channels(light_weight)

        # N,H,W,2
        img_weight = tf.expand_dims(light_weight[:,:,:,0], 3)
        pal_weight = tf.expand_dims(light_weight[:,:,:,1], 3)
        img_weight = tf.concat([img_weight, img_weight, img_weight], axis=3)
        pal_weight = tf.concat([pal_weight, pal_weight, pal_weight], axis=3)
        return input * img_weight + pal_weight

    def build_graph(self):
        with tf.variable_scope(self.name) as scope:
            # the input image
            self.input = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3), 'input')
            # the angle of the input image
            self.angle = tf.placeholder(tf.float32, (self.batch_size, self.angle_dim), 'angle')
            # the pair input image
            self.re_input = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3), 're_input')
            # the pair input image angle
            self.re_angle = tf.placeholder(tf.float32, (self.batch_size, self.angle_dim), 're_angle')

            # the flag of train or inference
            self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')

            with tf.variable_scope('angle_embedding'):
                # concat operation before angle embedding, we input the tow absolute angle of the pair images
                # angle_input = tf.concat([self.angle, self.re_angle], axis=1)
                angle_input = self.angle - self.re_angle
                # several dense layers
                angle_body = tf.layers.dense(angle_input, 16, name='dense1')
                angle_body = tf.nn.relu(angle_body, 'relu1')
                angle_body = tf.layers.dense(angle_body, 16, name='dense2')
                angle_body = tf.nn.relu(angle_body, 'relu2')
                # reshape the angle: N,16 --> N,1,1,16
                angle_body = tf.reshape(angle_body, shape=(-1, 1, 1, 16), name='reshape')
                # expansion the angle input: N,1,1,16 --> N,H,W,16
                angle_body = tf.tile(angle_body, (1, self.height, self.width, 1), name='tile')
                # assert?
                assert angle_body.get_shape().as_list()[1:] == [self.height, self.width, 16]

            # TODO:
            #   concat all the (image,landmark,angle) together as the input of coarse-warp
            #   this step is different from the original paper
            coarse_input = tf.concat([self.input, angle_body], axis=3, name='coarse_warp_concat')

            with tf.variable_scope('coarse_warp'):
                # the first layer
                coarse_body = self.conv2d_batch_norm_relu(coarse_input, 16, (5, 5), (1, 1), 'same', 'layer_1')
                # the second layer
                coarse_body = self.conv2d_batch_norm_relu(coarse_body, 32, (3, 3), (1, 1), 'same', 'layer_2')
                # down-sample
                coarse_body = tf.layers.average_pooling2d(coarse_body, (2, 2), (2, 2), 'valid', name='avg_pool_1')
                # the third layer
                coarse_body = light_feature_1 = self.conv2d_batch_norm_relu(coarse_body, 32, (3, 3), (1, 1), 'same', 'layer_3')
                # the forth layer
                coarse_body = self.conv2d_batch_norm_relu(coarse_body, 32, (3, 3), (1, 1), 'same', 'layer_4')
                # the final layer
                coarse_body = tf.layers.conv2d(coarse_body, 2, (1, 1), (1, 1), 'same', activation=tf.nn.tanh, name='layer_5')

                # coarse-fine pixel flow
                coarse_flow = tf.image.resize_images(coarse_body, (self.height, self.width), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # the coarse estimate
                coarse_image = self.bilinear_sample(self.input, coarse_flow, 'coarse_bilinear_sample')

            #   concat all the (image,coarse_image,pixel_flow) together as the input of fine-warp
            #   this step is different from the original paper
            fine_input = tf.concat([self.input, coarse_image, coarse_flow, angle_body], axis=3, name='fine_warp_concat')

            with tf.variable_scope('fine_warp'):
                # the first layer
                fine_body = self.conv2d_batch_norm_relu(fine_input, 16, (5, 5), (1, 1), 'same', 'layer_1')
                # the second layer
                fine_body = self.conv2d_batch_norm_relu(fine_body, 32, (3, 3), (1, 1), 'same', 'layer_2')
                # the third layer
                fine_body = light_feature_2 = self.conv2d_batch_norm_relu(fine_body, 32, (3, 3), (1, 1), 'same', 'layer_3')
                # the forth layer
                fine_body = self.conv2d_batch_norm_relu(fine_body, 32, (3, 3), (1, 1), 'same', 'layer_4')
                # the fifth layer
                fine_body = self.conv2d_batch_norm_relu(fine_body, 32, (3, 3), (1, 1), 'same', 'layer_5')
                # the final layer
                res_flow = tf.layers.conv2d(fine_body, 2, (1, 1), (1, 1), 'same', activation=tf.nn.tanh, name='res_flow')

                # add coarse-flow and fine-flow
                fine_flow = tf.add(coarse_flow, res_flow, name='fine_flow')
                # the fine estimate
                fine_image = self.bilinear_sample(self.input, fine_flow, 'fine_bilinear_sample')

            if self.light_on == True:
                with tf.variable_scope('light_module'):
                    # up-sample
                    light_feat_1_res = tf.image.resize_images(light_feature_1, (self.height, self.width),
                                                              tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    # light_feat_1_res = light_feature_1
                    # concat the coarse and fine part features
                    light_input = tf.concat([light_feat_1_res, light_feature_2], axis=3, name='light_input_concat')
                    # the first layer
                    light_body = self.conv2d_batch_norm_relu(light_input, 8, (1, 1), (1, 1), 'same', 'layer_1')
                    # the second layer
                    light_body = self.conv2d_batch_norm_relu(light_body, 8, (1, 1), (1, 1), 'same', 'layers_2')
                    # the final layer
                    light_body = tf.layers.conv2d(light_body, 2, (1, 1), (1, 1), 'same', name='layer_3')
                    # light-weight
                    self.output = self.pixel_light_weight(fine_image, light_body)
            else:
                self.output = fine_image

            with tf.name_scope('loss'):
                # for coarse image and loss
                batch_coarse_loss = tf.reduce_sum(tf.square(coarse_image - self.re_input), axis=(1, 2, 3))
                self.coarse_loss = tf.reduce_mean(batch_coarse_loss)
                # for fine image
                batch_fine_loss = tf.reduce_sum(tf.square(fine_image - self.re_input), axis=(1, 2, 3))
                self.fine_loss = tf.reduce_mean(batch_fine_loss)
                # for output image and loss
                batch_output_loss = tf.reduce_sum(tf.square(self.output - self.re_input), axis=(1, 2, 3))
                self.output_loss = tf.reduce_mean(batch_output_loss)
                # pixel loss
                self.pixel_loss = tf.reduce_mean(tf.square(self.output - self.re_input))
                # the final loss
                if self.light_on == True:
                    self.loss = self.coarse_loss +  self.fine_loss + self.output_loss
                else:
                    # output_loss --> self.fine_loss
                    self.loss = self.coarse_loss + self.output_loss

            # summary
            with tf.name_scope('summary'):
                image_max_outputs = 2
                tf.summary.image('input', self.input, max_outputs=image_max_outputs)
                tf.summary.image('re_input', self.re_input, max_outputs=image_max_outputs)
                tf.summary.image('coarse_image', coarse_image, max_outputs=image_max_outputs)
                tf.summary.image('fine_image', fine_image, max_outputs=image_max_outputs)
                tf.summary.histogram('coarse_flow_0_histogram', coarse_flow[:, :, :, 0])
                tf.summary.histogram('coarse_flow_1_histogram', coarse_flow[:, :, :, 1])
                tf.summary.histogram('fine_flow_0_histogram', fine_flow[:, :, :, 0])
                tf.summary.histogram('fine_flow_1_histogram', fine_flow[:, :, :, 1])
                tf.summary.image('coarse_flow_0', tf.expand_dims(coarse_flow[:,:,:,0], axis=3), max_outputs=image_max_outputs)
                tf.summary.image('coarse_flow_1', tf.expand_dims(coarse_flow[:,:,:,1], axis=3), max_outputs=image_max_outputs)
                tf.summary.image('fine_flow_0', tf.expand_dims(fine_flow[:,:,:,0], axis=3), max_outputs=image_max_outputs)
                tf.summary.image('fine_flow_1', tf.expand_dims(fine_flow[:,:,:,1], axis=3), max_outputs=image_max_outputs)
                tf.summary.image('output', self.output, max_outputs=image_max_outputs)
                tf.summary.scalar('fine_loss', self.fine_loss)
                tf.summary.scalar('coarse_loss', self.coarse_loss)
                tf.summary.scalar('output_loss', self.output_loss)
                tf.summary.scalar('pixel_loss', self.pixel_loss)
                tf.summary.histogram('batch_coarse_loss', batch_coarse_loss)
                tf.summary.histogram('batch_fine_loss', batch_fine_loss)
                tf.summary.histogram('batch_output_loss', batch_output_loss)

            # for restore at test time
            # all_variables = tf.trainable_variables()
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            self.angle_vars = [var for var in self.variables if 'angle_embedding' in var.name]
            self.coarse_vars = [var for var in self.variables if 'coarse_warp' in var.name]
            self.fine_vars = [var for var in self.variables if 'fine_warp' in var.name]



if __name__ == '__main__':
    gaze = DeepGaze(batch_size=64, name='deepgaze')
    gaze.build_graph()