# -*- coding: utf-8 -*-

'''
This is a stack GAN implementation of zi2zi.  There is 2 identical (except for size) networks that first generate a small image, 
then use that small image and the initial input to generate a larger image
This has shown promising results on generating larger images, so it is worth a try on english word decoding, where the images are larger
'''
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple
from .ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding, conditional_instance_norm
from .dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider
from .utils import scale_back, merge, save_concat_images
from math import ceil



# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss", "const_loss", "l1_loss",
                                       "category_loss", "cheat_loss", "tv_loss"])
IntermediateLossHandle = namedtuple("InterLossHandle", ["d_loss", "g_loss", "const_loss", "l1_loss",
                                       "category_loss", "cheat_loss", "tv_loss"])

InputHandle = namedtuple("InputHandle", ["real_data", "embedding_ids", "no_target_data", "no_target_ids"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source", "embedding"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])


class SNet(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=1024,input_height=256, output_width=1024, output_height=256,
                intermediate_width=256,intermediate_height=64, intermediate_gen_dim=4, intermediate_dis_dim=4,
                generator_dim=64, discriminator_dim=64, L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,
                Lcategory_penalty=1.0, embedding_num=40, embedding_dim=128, input_filters=3, output_filters=3):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
	    self.input_height = input_height
        self.output_width = output_width
	    self.output_height = output_height
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.intermediate_width = intermediate_width
        self.intermediate_height = intermediate_height
        self.intermediate_gen_dim = intermediate_gen_dim
        self.intermediate_dis_dim = intermediate_dis_dim
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.input_filters = input_filters
        self.output_filters = output_filters
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")
    def inter_encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator1"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="gi_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="gi_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc
            #to do: try not collapsing final axis?
            #initial dim:256x64
            e1 = conv2d(images, self.generator_dim, scope="gi_e1_conv") #128x32
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2) #64x16
            e3 = encode_layer(e2, self.generator_dim * 4, 3) #32x8
            e4 = encode_layer(e3, self.generator_dim * 8, 4) #16x4
            e5 = encode_layer(e4, self.generator_dim * 8, 5) #8x2
            e6 = encode_layer(e5, self.generator_dim * 8, 6) #4x1  <-- should i stop here?
            # e7 = encode_layer(e6, self.generator_dim * 8, 7) #2x1
            # e8 = encode_layer(e7, self.generator_dim * 8, 8) #1x1
            return e6, encode_layers
    
    def inter_decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("generator1"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
     
            s = self.intermediate_width

            s2, s4, s8, s16, s32 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32)

            sh = self.intermediate_height

            sh2, sh4, sh8, sh16, sh32 = int(sh / 2), int(sh / 4), int(sh / 8), int(sh / 16), int(sh / 32)

            def decode_layer(x, output_height, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_height,
                                                   output_width, output_filters], scope="gi_d%d_deconv" % layer)
                
                if layer != 6:
                        # IMPORTANT: normalization for last layer
                        # Very important, otherwise GAN is unstable
                        # Trying conditional instance normalization to
                        # overcome the fact that batch normalization offers
                        # different train/test statistics
                        if inst_norm:
                            dec = conditional_instance_norm(dec, ids, self.embedding_num, scope="gi_d%d_inst_norm" % layer)
                        else:
                            dec = batch_norm(dec, is_training, scope="gi_d%d_bn" % layer)
                    if dropout:
                        dec = tf.nn.dropout(dec, 0.5)
                    if do_concat:
                        dec = tf.concat([dec, enc_layer], 3)
                    return dec

            #should start out 4x1
            d1 = decode_layer(encoded, sh32, s32, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e6"],
                              dropout=True) #8x2
            d2 = decode_layer(d1, sh16, s16, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e5"], dropout=True) #16x4     
            d3 = decode_layer(d2, sh8, s8, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e4"], dropout=True) #32x8
            d4 = decode_layer(d3, sh4, s4, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e3"], dropout=True) #64x16
            d5 = decode_layer(d4, sh2, s2, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e2"], dropout=True) #128x32
            d6 = decode_layer(d5, sh, s, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e1"]) #256x64
            
            output = tf.nn.tanh(d6)  # scale to (-1, 1)
            return output


    def encode_inter(self, images, is_training, reuse=False):
        with tf.variable_scope("generator1"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc
            #to do: try not collapsing final axis?
            #initial dim:256x16
            #encode down to 64/16
            e1 = conv2d(images, self.generator_dim*2, scope="g_e1_conv") 
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 4, 2)

            return e2, encode_layers


    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc
            #original: 1024x256
            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv") #512x128
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2) #256x64
            e3 = encode_layer(e2, self.generator_dim * 4, 3) #128x32
            e4 = encode_layer(e3, self.generator_dim * 4, 4) #64x16
            return e4, encode_layers

    def decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
	    
            s = self.output_width
            s2, s4, s8 = int(s / 2), int(s / 4), int(s / 8)
            
            sh = self.output_height
            sh2, sh4, sh8 = int(sh / 2), int(sh / 4), int(sh / 8),
            
            def decode_layer(x, output_height, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=False):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_height,
                                                   output_width, output_filters], scope="g_d%d_deconv" % layer)
                
                if layer != 4:
                        # IMPORTANT: normalization for last layer
                        # Very important, otherwise GAN is unstable
                        # Trying conditional instance normalization to
                        # overcome the fact that batch normalization offers
                        # different train/test statistics
                        if inst_norm:
                            dec = conditional_instance_norm(dec, ids, self.embedding_num, scope="g_d%d_inst_norm" % layer)
                        else:
                            dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                    if dropout:
                        dec = tf.nn.dropout(dec, 0.5)
                    if do_concat:
                        dec = tf.concat([dec, enc_layer], 3)
                    return dec

            #start: 64/16
            d1 = decode_layer(encoded, sh8, s8, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e4"],
                              dropout=True) #128x32
            d2 = decode_layer(d1, sh4, s4, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e3"], dropout=True)  #256x64    
            d3 = decode_layer(d2, sh2, s2, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e2"], dropout=True) #512x128
            d4 = decode_layer(d3, sh, s, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e1"], dropout=True) #1024x256
            
            output = tf.nn.tanh(d4)  # scale to (-1, 1)
            return output

    def generator(self, images, inter_images, embeddings, embedding_ids, inst_norm, is_training, reuse=False):
        i8, ienc_layers = self.inter_encoder(inter_images, is_training=is_training, reuse=reuse)
        local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
        local_embeddings = tf.reshape(local_embeddings, [self.batch_size, 1, 1, self.embedding_dim])
        embedded = tf.concat([e8, local_embeddings], 3)
        inter_output = self.inter_decoder(embedded, enc_layers, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)

        ie8, enc_layers = self.encode_inter(inter_output, is_training=is_training, reuse=reuse)
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
        embedded = tf.concat([e8, ie8, local_embeddings], 3)
        output = self.decoder(embedded, enc_layers, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)
        return output, e8, inter_output, ie8

    def discriminator(self, image, is_training, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")
            # category loss
            fc2 = fc(tf.reshape(h3, [self.batch_size, -1]), self.embedding_num, scope="d_fc2")
            return tf.nn.sigmoid(fc1), fc1, fc2

    #need to standardize names of intermediate step, this is a mess
    def build_model(self, is_training=True, inst_norm=False, no_target_source=False):
        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_height, self.input_width,
                                    self.input_filters + self.output_filters],
                                   name='real_A_and_B_images')
        real_shrunk_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.intermediate_height, self.intermediate_height,
                                    self.input_filters + self.output_filters],
                                   name='shrunk_real_A_and_B_images')
        embedding_ids = tf.placeholder(tf.int64, shape=None, name="embedding_ids")
        no_target_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.input_height, self.input_width,
                                         self.input_filters + self.output_filters],
                                        name='no_target_A_and_B_images')
        no_target_ids = tf.placeholder(tf.int64, shape=None, name="no_target_embedding_ids")


        # target images
        real_B = real_data[:, :, :, :self.input_filters]
        # source images
        real_A = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]

        #intermediate target images
        inter_B = real_shrunk_data[:, :, :, :self.input_filters]
        #intermediate source images
        inter_A = real_shrunk_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]

        embedding = init_embedding(self.embedding_num, self.embedding_dim)
        fake_B, encoded_real_A, fake_inter_B, encoded_inter_A = self.generator(real_A, inter_A, embedding, embedding_ids, is_training=is_training,
                                                inst_norm=inst_norm)
        real_final_AB = tf.concat([real_A, real_B], 3)
        fake_final_AB = tf.concat([real_A, fake_B], 3)

        final_width = self.output_width
        final_height = self.output_height

        real_inter_AB = tf.concat([inter_A, inter_B], 3)
        fake_inter_AB = tf.concat([inter_A, inter_B], 3)

        inter_width = self.intermediate_width
        inter_height = self.intermediate_height

        def build_loss_handle(real_AB, fake_AB, width, height):
            # Note it is not possible to set reuse flag back to False
            # initialize all variables before setting reuse to True
            real_D, real_D_logits, real_category_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
            fake_D, fake_D_logits, fake_category_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)

            # encoding constant loss
            # this loss assume that generated imaged and real image
            # should reside in the same space and close to each other
            encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
            const_loss = (tf.reduce_mean(tf.square(encoded_real_A - encoded_fake_B))) * self.Lconst_penalty

            # category loss
            true_labels = tf.reshape(tf.one_hot(indices=embedding_ids, depth=self.embedding_num),
                                     shape=[self.batch_size, self.embedding_num])
            real_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_category_logits,
                                                                                        labels=true_labels))
            fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_logits,
                                                                                        labels=true_labels))
            category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

            # binary real/fake loss
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                                 labels=tf.ones_like(real_D)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                                 labels=tf.zeros_like(fake_D)))
            # L1 loss between real and generated images
            l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
            # total variation loss
            tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :height - 1, :, :]) / height
                       + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width) * self.Ltv_penalty

            # maximize the chance generator fool the discriminator
            cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                                labels=tf.ones_like(fake_D)))

            d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
            g_loss = cheat_loss + l1_loss + self.Lcategory_penalty * fake_category_loss + const_loss + tv_loss
            if no_target_source:
                # no_target source are examples that don't have the corresponding target images
                # however, except L1 loss, we can compute category loss, binary loss and constant losses with those examples
                # it is useful when discriminator get saturated and d_loss drops to near zero
                # those data could be used as additional source of losses to break the saturation
                no_target_A = no_target_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]
                no_target_B, encoded_no_target_A = self.generator(no_target_A, embedding, no_target_ids,
                                                                  is_training=is_training,
                                                                  inst_norm=inst_norm, reuse=True)
                no_target_labels = tf.reshape(tf.one_hot(indices=no_target_ids, depth=self.embedding_num),
                                              shape=[self.batch_size, self.embedding_num])
                no_target_AB = tf.concat([no_target_A, no_target_B], 3)
                no_target_D, no_target_D_logits, no_target_category_logits = self.discriminator(no_target_AB,
                                                                                                is_training=is_training,
                                                                                                reuse=True)
                encoded_no_target_B = self.encoder(no_target_B, is_training, reuse=True)[0]
                no_target_const_loss = tf.reduce_mean(
                    tf.square(encoded_no_target_A - encoded_no_target_B)) * self.Lconst_penalty
                no_target_category_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_category_logits,
                                                            labels=no_target_labels)) * self.Lcategory_penalty

                d_loss_no_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_D_logits,
                                                                                          labels=tf.zeros_like(
                                                                                              no_target_D)))
                cheat_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_target_D_logits,
                                                                                     labels=tf.ones_like(no_target_D)))
                d_loss = d_loss_real + d_loss_fake + d_loss_no_target + (category_loss + no_target_category_loss) / 3.0
                g_loss = cheat_loss / 2.0 + l1_loss + \
                         (self.Lcategory_penalty * fake_category_loss + no_target_category_loss) / 2.0 + \
                         (const_loss + no_target_const_loss) / 2.0 + tv_loss

            d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
            d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)
            category_loss_summary = tf.summary.scalar("category_loss", category_loss)
            cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss)
            l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
            fake_category_loss_summary = tf.summary.scalar("fake_category_loss", fake_category_loss)
            const_loss_summary = tf.summary.scalar("const_loss", const_loss)
            d_loss_summary = tf.summary.scalar("d_loss", d_loss)
            g_loss_summary = tf.summary.scalar("g_loss", g_loss)
            tv_loss_summary = tf.summary.scalar("tv_loss", tv_loss)

            d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
                                                 category_loss_summary, d_loss_summary])
            g_merged_summary = tf.summary.merge([cheat_loss_summary, l1_loss_summary,
                                                 fake_category_loss_summary,
                                                 const_loss_summary,
                                                 g_loss_summary, tv_loss_summary])

            loss_handle = LossHandle(d_loss=d_loss,
                                     g_loss=g_loss,
                                     const_loss=const_loss,
                                     l1_loss=l1_loss,
                                     category_loss=category_loss,
                                     cheat_loss=cheat_loss,
                                     tv_loss=tv_loss)
            summary_handle = SummaryHandle(d_merged=d_merged_summary,
                                           g_merged=g_merged_summary)
            return loss_handle, summary_handle

        loss_handle, summary_handle = build_loss_handle(real_final_AB, fake_final_AB, final_width, final_height)
        inter_loss_handle, inter_summary_handle = build_loss_handle(real_inter_AB, fake_inter_AB, inter_width, inter_height)
        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data,
                                   embedding_ids=embedding_ids,
                                   no_target_data=no_target_data,
                                   no_target_ids=no_target_ids)
        eval_handle = EvalHandle(encoder=encoded_real_A,
                                 generator=fake_B,
                                 target=real_B,
                                 source=real_A,
                                 embedding=embedding)
        inter_input_handle = InputHandle(real_data=real_shrunk_data,
                                   embedding_ids=embedding_ids,
                                   no_target_data=no_target_data,
                                   no_target_ids=no_target_ids)

        inter_eval_handle = EvalHandle(encoder=encoded_inter_A,
                                 generator=fake_inter_B,
                                 target=inter_B,
                                 source=inter_A,
                                 embedding=embedding)

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

        setattr(self, "inter_input_handle", inter_input_handle)
        setattr(self, "inter_loss_handle", inter_loss_handle)
        setattr(self, "inter_eval_handle", inter_eval_handle)
        setattr(self, "inter_summary_handle", inter_summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        di_vars = [var for var in t_vars if 'di_' in var.name]
        gi_vars = [var for var in t_vars if 'gi_' in var.name]
        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e" in var.name)]
            gi_vars = [var for var in g_vars if not ("gi_e" in var.name)]
        return g_vars, d_vars

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if 'embedding' in var.name or "g_" in var.name or "g_i" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle

    def retrieve_inter_handles(self):
        input_handle = getattr(self, "inter_input_handle")
        loss_handle = getattr(self, "inter_loss_handle")
        eval_handle = getattr(self, "inter_eval_handle")
        summary_handle = getattr(self, "inter_summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle


    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "snet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images, shrunk_images, embedding_ids, loss_to_use = 'inter'):
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handles()
        inter_input_handle, inter_loss_handle, inter_eval_handle, inter_summary_handle = self.retrieve_inter_handles()
        if loss_to_use == 'inter':
            loss_handle = inter_loss_handle
        fake_images, real_images, \
        fake_inter_images, real_inter_images, \
        d_loss, g_loss, l1_loss = self.sess.run([eval_handle.generator,
                                                 eval_handle.target,
                                                 inter_eval_handle.generator,
                                                 inter_eval_handle.target,
                                                 loss_handle.d_loss,
                                                 loss_handle.g_loss,
                                                 loss_handle.l1_loss],
                                                feed_dict={
                                                    input_handle.real_data: input_images,
                                                    input_handle.embedding_ids: embedding_ids,
                                                    input_handle.no_target_data: input_images,
                                                    input_handle.no_target_ids: embedding_ids,
                                                    inter_input_handle.real_data: shrunk_images,
                                                    inter_input_handle.embedding_ids: embedding_ids,
                                                    inter_input_handle.no_target_data: shrunk_images,
                                                    inter_input_handle.no_target_ids: embedding_ids,


                                                })
        return fake_images, real_images, fake_inter_images, real_inter_images, d_loss, g_loss, l1_loss

    def validate_model(self, val_iter, epoch, step, loss_to_use = 'inter'):
        labels, images, shrunk_images = next(val_iter)
        fake_imgs, real_imgs, fake_inter_imgs, real_inter_imgs, d_loss, g_loss, l1_loss = self.generate_fake_samples(images, shrunk_images, labels, loss_to_use)
        print("Sample: d_loss: %.5f, g_loss: %.5f, l1_loss: %.5f" % (d_loss, g_loss, l1_loss))

        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        merged_inter_fake = merge(scale_back(fake_inter_imgs), [self.batch_size, 1])
        merged_inter_real = merge(scale_back(real_inter_imgs), [self.batch_size, 1])
        merged_inter_pair = concatenate([merged_inter_real, merged_inter_fake], axis=1)

        merged_pair = concatenate([merged_inter_pair, merged_pair], axis=1)

        model_id, _ = self.get_model_id_and_dir()

        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, embedding_ids, model_dir, save_dir):
        source_provider = InjectDataProvider(source_obj)

        if isinstance(embedding_ids, int) or len(embedding_ids) == 1:
            embedding_id = embedding_ids if isinstance(embedding_ids, int) else embedding_ids[0]
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, embedding_id)
        else:
            source_iter = source_provider.get_random_embedding_iter(self.batch_size, embedding_ids)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count, real_imgs=None):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
	    print("generated images saved at %s" % p)
	    if real_imgs:
		p = os.path.join(save_dir, "real_%04d.png" % count)
		save_concat_images(real_imgs, img_path=p)
		print("real images saved at %s" % p) 
            
        count = 0
        batch_buffer = list()
	    real_batch_buffer = list()
        for labels, source_imgs in source_iter:
            fake_imgs, real_imgs = self.generate_fake_samples(source_imgs, labels)[:2]
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
	    batch_buffer.append(merged_fake_images)
	    real_batch_buffer.append(merged_real_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count, real_batch_buffer)
                batch_buffer = list()
		real_batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count, real_batch_buffer)

    def interpolate(self, source_obj, between, model_dir, save_dir, steps):
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)
        # new interpolated dimension
        new_x_dim = steps + 1
        alphas = np.linspace(0.0, 1.0, new_x_dim)

        def _interpolate_tensor(_tensor):
            """
            Compute the interpolated tensor here
            """

            x = _tensor[between[0]]
            y = _tensor[between[1]]

            interpolated = list()
            for alpha in alphas:
                interpolated.append(x * (1. - alpha) + alpha * y)

            interpolated = np.asarray(interpolated, dtype=np.float32)
            return interpolated

        def filter_embedding_vars(var):
            var_name = var.name
            if var_name.find("embedding") != -1:
                return True
            if var_name.find("inst_norm/shift") != -1 or var_name.find("inst_norm/scale") != -1:
                return True
            return False

        embedding_vars = filter(filter_embedding_vars, tf.trainable_variables())
        # here comes the hack, we overwrite the original tensor
        # with interpolated ones. Note, the shape might differ

        # this is to restore the embedding at the end
        embedding_snapshot = list()
        for e_var in embedding_vars:
            val = e_var.eval(session=self.sess)
            embedding_snapshot.append((e_var, val))
            t = _interpolate_tensor(val)
            op = tf.assign(e_var, t, validate_shape=False)
            print("overwrite %s tensor" % e_var.name, "old_shape ->", e_var.get_shape(), "new shape ->", t.shape)
            self.sess.run(op)

        source_provider = InjectDataProvider(source_obj)
        input_handle, _, eval_handle, _ = self.retrieve_handles()
        for step_idx in range(len(alphas)):
            alpha = alphas[step_idx]
            print("interpolate %d -> %.4f + %d -> %.4f" % (between[0], 1. - alpha, between[1], alpha))
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, 0)
            batch_buffer = list()
            count = 0
            for _, source_imgs in source_iter:
                count += 1
                labels = [step_idx] * self.batch_size
                generated, = self.sess.run([eval_handle.generator],
                                           feed_dict={
                                               input_handle.real_data: source_imgs,
                                               input_handle.embedding_ids: labels
                                           })
                merged_fake_images = merge(scale_back(generated), [self.batch_size, 1])
                batch_buffer.append(merged_fake_images)
            if len(batch_buffer):
                save_concat_images(batch_buffer,
                                   os.path.join(save_dir, "frame_%02d_%02d_step_%02d.png" % (
                                       between[0], between[1], step_idx)))
        # restore the embedding variables
        print("restore embedding values")
        for var, val in embedding_snapshot:
            op = tf.assign(var, val, validate_shape=False)
            self.sess.run(op)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, flip_labels=False,
              freeze_encoder=False, fine_tune=None, sample_steps=50, checkpoint_steps=500):
        g_vars, d_vars, gi_vars, di_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)

        input_handle, loss_handle, _, summary_handle = self.retrieve_handles()
        inter_input_handle, inter_loss_handle, inter_eval_handle, inter_summary_handle = self.retrieve_inter_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        tf.global_variables_initializer().run()
        real_data = input_handle.real_data
        shrunk_data = inter_input_handle.real_data
        embedding_ids = input_handle.embedding_ids
        no_target_data = input_handle.no_target_data
        no_target_ids = input_handle.no_target_ids

        # filter by one type of labels
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune, shrink=True)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val_iter(self.batch_size)

        saver = tf.train.Saver(max_to_keep=3)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        def run_against_loss(loss_handle, summary_handle, loss_to_use='inter'):
            d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
            g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)
            for ei in range(epoch):
                train_batch_iter = data_provider.get_train_iter(self.batch_size)

                if (ei + 1) % schedule == 0:
                    update_lr = current_lr / 2.0
                    # minimum learning rate guarantee
                    update_lr = max(update_lr, 0.0002)
                    print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                    current_lr = update_lr

                for bid, batch in enumerate(train_batch_iter):
                    counter += 1
                    labels, batch_images, shrunk_data = batch
                    shuffled_ids = labels[:]
                    if flip_labels:
                        np.random.shuffle(shuffled_ids)
                    # Optimize D
                    _, batch_d_loss, d_summary = self.sess.run([d_optimizer, loss_handle.d_loss,
                                                                summary_handle.d_merged],
                                                               feed_dict={
                                                                   real_data: batch_images,
                                                                   real_shrunk_data:shrunk_data
                                                                   embedding_ids: labels,
                                                                   learning_rate: current_lr,
                                                                   no_target_data: batch_images,
                                                                   no_target_ids: shuffled_ids
                                                               })
                    # Optimize G
                    _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                    feed_dict={
                                                        real_data: batch_images,
                                                        real_shrunk_data:shrunk_data
                                                        embedding_ids: labels,
                                                        learning_rate: current_lr,
                                                        no_target_data: batch_images,
                                                        no_target_ids: shuffled_ids
                                                    })
                    # magic move to Optimize G again
                    # according to https://github.com/carpedm20/DCGAN-tensorflow
                    # collect all the losses along the way
                    _, batch_g_loss, category_loss, cheat_loss, \
                    const_loss, l1_loss, tv_loss, g_summary = self.sess.run([g_optimizer,
                                                                             loss_handle.g_loss,
                                                                             loss_handle.category_loss,
                                                                             loss_handle.cheat_loss,
                                                                             loss_handle.const_loss,
                                                                             loss_handle.l1_loss,
                                                                             loss_handle.tv_loss,
                                                                             summary_handle.g_merged],
                                                                            feed_dict={
                                                                                real_data: batch_images,
                                                                                real_shrunk_data:shrunk_data
                                                                                embedding_ids: labels,
                                                                                learning_rate: current_lr,
                                                                                no_target_data: batch_images,
                                                                                no_target_ids: shuffled_ids
                                                                            })
                    passed = time.time() - start_time
                    log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, " + \
                                 "category_loss: %.5f, cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f, tv_loss: %.5f"
                    print(log_format % (ei, bid, total_batches, passed, batch_d_loss, batch_g_loss,
                                        category_loss, cheat_loss, const_loss, l1_loss, tv_loss))
                    summary_writer.add_summary(d_summary, counter)
                    summary_writer.add_summary(g_summary, counter)

                    if counter % sample_steps == 0:
                        # sample the current model states with val data
                        self.validate_model(val_batch_iter, ei, counter, loss_to_use=loss_to_use)

                    if counter % checkpoint_steps == 0:
                        print("Checkpoint: save checkpoint step %d" % counter)
                        self.checkpoint(saver, counter)

            # save the last checkpoint
            print("Checkpoint: last checkpoint step %d" % counter)
            self.checkpoint(saver, counter)
        
        run_against_loss(inter_loss_handle, inter_summary_handle, loss_to_use== 'inter')
        run_against_loss(loss_handle, summary_handle, loss_to_use== 'final')
