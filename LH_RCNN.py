from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
import numpy as np


class LHRCNN:
    def __init__(self, config, data_provider):

        assert len(config['data_shape']) == 3
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.is_pretraining = config['is_pretraining']
        self.data_shape = config['data_shape']
        if not self.is_pretraining:
            self.num_classes = config['num_classes'] + 1
        else:
            self.num_classes = config['num_classes']
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']

        self.anchor_scales = [32, 64, 128, 256, 512]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        self.post_nms_proposals = 2000 if self.mode == 'train' else 1000

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator

        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True

        if self.is_pretraining:
            self._define_pretraining_inputs()
            self._build_pretraining_graph()
            self._create_pretraining_saver()
            self.save_weight = self._save_pretraining_weight
            self.train_one_epoch = self._train_pretraining_epoch
            self.test_one_image = self._test_one_pretraining_image
            if self.mode == 'train':
                self._create_pretraining_summary()
        else:
            self._define_detection_inputs()
            self._build_detection_graph()
            self._create_detection_saver()
            self.save_weight = self._save_detection_weight
            self.train_one_epoch = self._train_detection_epoch
            self.test_one_image = self._test_one_detection_image
            if self.mode == 'train':
                self._create_detection_summary()
        self._init_session()

    def _define_pretraining_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)
        if self.data_format == 'channels_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])
        if self.mode == 'train':
            self.images, self.labels = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images - mean
            self.labels = tf.cast(self.labels, tf.int32)
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images - mean
            self.labels = tf.placeholder(tf.int32, [self.batch_size], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _define_detection_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)
        if self.data_format == 'channels_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images - mean
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images - mean
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, None, 5], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_pretraining_graph(self):
        with tf.variable_scope('feature_extractor'):
            features, _ = self._feature_extractor(self.images)
        with tf.variable_scope('pretraining'):
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(features, axis=axes, name='global_pool')
            final_dense = tf.layers.dense(global_pool, self.num_classes)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=final_dense, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(final_dense, 1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.pred, tf.cast(self.labels, tf.int64)), tf.float32), name='accuracy'
            )
            self.loss = loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
            ) + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('pretraining')]
            )
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_detection_graph(self):
        with tf.variable_scope('feature_extractor'):
            c4_feat, downsampling_rate = self._feature_extractor(self.images)
        with tf.variable_scope('stage5'):
            state5_conv1_1 = self._separable_conv_layer(c4_feat, 256, [1, 15], 1, 'state5_conv1_1', activation=tf.nn.relu)
            state5_conv1_2 = self._separable_conv_layer(state5_conv1_1, 490, [15, 1], 1, 'state5_conv1_2', activation=tf.nn.relu)
            state5_conv2_1 = self._separable_conv_layer(c4_feat, 256, [1, 15], 1, 'state5_conv2_1', activation=tf.nn.relu)
            state5_conv2_2 = self._separable_conv_layer(state5_conv2_1, 490, [15, 1], 1, 'state5_conv2_2', activation=tf.nn.relu)
            rcnn_feat = state5_conv1_2 + state5_conv2_2
            pshape = tf.shape(rcnn_feat)

        with tf.variable_scope('RPN'):
            scale = tf.constant([downsampling_rate, downsampling_rate, downsampling_rate, downsampling_rate, 1], dtype=tf.float32)
            scale = tf.reshape(scale, [1, 1, 5])
            nground_truth = self.ground_truth / scale
            rpn_conv = self._conv_layer(c4_feat, 256, 3, 1, activation=tf.nn.relu)
            rpn_pbbox = self._conv_layer(rpn_conv, self.num_anchors * 4, 1, 1)
            rpn_pconf = self._conv_layer(rpn_conv, self.num_anchors * 2, 1, 1)
            rpn_pbbox = tf.reshape(rpn_pbbox, [self.batch_size, -1, 4])
            rpn_pconf = tf.reshape(rpn_pconf, [self.batch_size, -1, 2])
            rpn_pbbox_yx = rpn_pbbox[..., :2]
            rpn_pbbox_hw = rpn_pbbox[..., 2:]
            abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw = self._get_abbox(pshape, downsampling_rate)
            if self.mode == 'train':
                min_mask = tf.cast(abbox_y1x1[:, 0] >= 0., tf.float32) * tf.cast(abbox_y1x1[:, 1] >= 0., tf.float32)
                max_mask = tf.cast(abbox_y2x2[:, 0] <= tf.cast(pshape[1]-1, tf.float32), tf.float32) * tf.cast(abbox_y2x2[:, 1] <= tf.cast(pshape[2]-1, tf.float32), tf.float32)
                mask = tf.reshape((min_mask * max_mask) > 0., [-1])
                abbox_y1x1 = tf.boolean_mask(abbox_y1x1, mask)
                abbox_y2x2 = tf.boolean_mask(abbox_y2x2, mask)
                abbox_yx = tf.boolean_mask(abbox_yx, mask)
                abbox_hw = tf.boolean_mask(abbox_hw, mask)
                rpn_pbbox_yx = tf.boolean_mask(rpn_pbbox_yx, mask, axis=1)
                rpn_pbbox_hw = tf.boolean_mask(rpn_pbbox_hw, mask, axis=1)
                rpn_pconf = tf.boolean_mask(rpn_pconf, mask, axis=1)
        with tf.variable_scope('RCNN'):
            if self.mode == 'train':
                rpn_loss = []
                proposal = []
                proposal_yx = []
                proposal_hw = []
                pos_proposal_ngyx = []
                pos_proposal_nghw = []
                box_ind = []
                rcnn_label = []
                for i in range(self.batch_size):
                    rpn_loss_, proposal_, proposal_yx_, proposal_hw_, pos_proposal_ngyx_, pos_proposal_nghw_, rcnn_label_ = \
                        self._one_image_rpn_train(rpn_pconf[i, ...], rpn_pbbox_yx[i, ...], rpn_pbbox_hw[i, ...],
                                                  abbox_yx, abbox_hw, abbox_y1x1, abbox_y2x2, nground_truth[i, ...], pshape)
                    box_ind_ = tf.reshape(tf.zeros_like(proposal_[:, 0], dtype=tf.int32) + i, [-1])
                    rpn_loss.append(rpn_loss_)
                    proposal.append(proposal_)
                    proposal_yx.append(proposal_yx_)
                    proposal_hw.append(proposal_hw_)
                    pos_proposal_ngyx.append(pos_proposal_ngyx_)
                    pos_proposal_nghw.append(pos_proposal_nghw_)
                    box_ind.append(box_ind_)
                    rcnn_label.append(rcnn_label_)
                rpn_loss = tf.reduce_mean(rpn_loss)
                proposal = tf.concat(proposal, axis=0)
                proposal_yx = tf.concat(proposal_yx, axis=0)
                proposal_hw = tf.concat(proposal_hw, axis=0)
                pos_proposal_ngyx = tf.concat(pos_proposal_ngyx, axis=0)
                pos_proposal_nghw = tf.concat(pos_proposal_nghw, axis=0)
                box_ind = tf.concat(box_ind, axis=0)
                rcnn_label = tf.concat(rcnn_label, axis=0)
            else:
                proposal_yx, proposal_hw, proposal = self._one_image_rpn_test(
                    rpn_pconf[0, ...], rpn_pbbox_yx[0, ...], rpn_pbbox_hw[0, ...], abbox_yx, abbox_hw, pshape)
                box_ind = tf.reshape(tf.zeros_like(proposal[:, 0], dtype=tf.int32) + 0, [-1])

            roi_feat = tf.image.crop_and_resize(rcnn_feat, proposal, box_ind, [7, 7])
            roi_feat = tf.layers.flatten(roi_feat)
            rcnn_conf = tf.layers.dense(roi_feat, self.num_classes)
            rcnn_pbbox = tf.layers.dense(roi_feat, 4)
            rcnn_pbbox_yx = rcnn_pbbox[:, :2]
            rcnn_pbbox_hw = rcnn_pbbox[:, 2:]

            if self.mode == 'train':
                rcnn_label_mask = rcnn_label < self.num_classes - 1
                rcnn_pbbox_yx = tf.boolean_mask(rcnn_pbbox_yx, rcnn_label_mask)
                rcnn_pbbox_hw = tf.boolean_mask(rcnn_pbbox_hw, rcnn_label_mask)
                proposal_yx = tf.boolean_mask(proposal_yx, rcnn_label_mask)
                proposal_hw = tf.boolean_mask(proposal_hw, rcnn_label_mask)
                rcnn_gbbox_yx = (pos_proposal_ngyx - proposal_yx) / proposal_hw
                rcnn_gbbox_hw = tf.log(pos_proposal_nghw / proposal_hw)
                bbox_yx_loss = self._smooth_l1_loss(rcnn_gbbox_yx - rcnn_pbbox_yx)
                bbox_hw_loss = self._smooth_l1_loss(rcnn_gbbox_hw - rcnn_pbbox_hw)
                bbox_loss = tf.reduce_mean(bbox_yx_loss + bbox_hw_loss)
                conf_loss = tf.losses.sparse_softmax_cross_entropy(labels=rcnn_label, logits=rcnn_conf, reduction=tf.losses.Reduction.MEAN)
                total_loss = rpn_loss + conf_loss + bbox_loss

                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
                self.loss = total_loss + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
                )
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                bbox_yx = rcnn_pbbox_yx * proposal_hw + proposal_yx
                bbox_hw = proposal_hw * tf.exp(rcnn_pbbox_hw)
                bbox_y1x1 = bbox_yx - bbox_hw / 2.
                bbox_y2x2 = bbox_yx + bbox_hw / 2.
                confidence = tf.nn.softmax(rcnn_conf)
                class_id = tf.argmax(confidence, axis=-1)
                conf_mask = tf.less(class_id, self.num_classes - 1)
                bbox_y1x1y2x2 = tf.concat([bbox_y1x1, bbox_y2x2], axis=-1)
                bbox_y1x1y2x2 = tf.boolean_mask(bbox_y1x1y2x2, conf_mask) * downsampling_rate
                confidence = tf.boolean_mask(confidence, conf_mask)[:, :self.num_classes - 1]
                filter_mask = tf.greater_equal(confidence, self.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(self.num_classes - 1):
                    scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                    bboxi = tf.boolean_mask(bbox_y1x1y2x2, filter_mask[:, i])
                    selected_indices = tf.image.non_max_suppression(

                        bboxi, scoresi, self.nms_max_boxes, self.nms_iou_threshold,
                    )
                    scores.append(tf.gather(scoresi, selected_indices))
                    bbox.append(tf.gather(bboxi, selected_indices))
                    class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
                bbox = tf.concat(bbox, axis=0)
                scores = tf.concat(scores, axis=0)
                class_id = tf.concat(class_id, axis=0)
                self.detection_pred = [scores, bbox, class_id]

    def _feature_extractor(self, image):
        with tf.variable_scope('stage1'):
            conv1 = self._conv_layer(image, 24, 3, 2, 'conv1', activation=tf.nn.relu)
            pool1 = self._max_pooling(conv1, 3, 2, 'pool1')
        with tf.variable_scope('stage2'):
            stage2_sconv1 = self._separable_conv_layer(pool1, 144, 3, 2, 'stage2_sconv1', activation=tf.nn.relu)
            stage2_sconv2 = self._separable_conv_layer(stage2_sconv1, 144, 3, 1, 'stage2_sconv2', activation=tf.nn.relu)
            stage2_sconv3 = self._separable_conv_layer(stage2_sconv2, 144, 3, 1, 'stage2_sconv3', activation=tf.nn.relu)
            stage2_sconv4 = self._separable_conv_layer(stage2_sconv3, 144, 3, 1, 'stage2_sconv4', activation=tf.nn.relu)
        with tf.variable_scope('stage3'):
            stage3_sconv1 = self._separable_conv_layer(stage2_sconv4, 288, 3, 2, 'stage3_sconv1', activation=tf.nn.relu)
            stage3_sconv2 = self._separable_conv_layer(stage3_sconv1, 288, 3, 1, 'stage3_sconv2', activation=tf.nn.relu)
            stage3_sconv3 = self._separable_conv_layer(stage3_sconv2, 288, 3, 1, 'stage3_sconv3', activation=tf.nn.relu)
            stage3_sconv4 = self._separable_conv_layer(stage3_sconv3, 288, 3, 1, 'stage3_sconv4', activation=tf.nn.relu)
            stage3_sconv5 = self._separable_conv_layer(stage3_sconv4, 288, 3, 1, 'stage3_sconv5', activation=tf.nn.relu)
            stage3_sconv6 = self._separable_conv_layer(stage3_sconv5, 288, 3, 1, 'stage3_sconv6', activation=tf.nn.relu)
            stage3_sconv7 = self._separable_conv_layer(stage3_sconv6, 288, 3, 1, 'stage3_sconv7', activation=tf.nn.relu)
            stage3_sconv8 = self._separable_conv_layer(stage3_sconv7, 288, 3, 1, 'stage3_sconv8', activation=tf.nn.relu)
        with tf.variable_scope('stage4'):
            stage4_sconv1 = self._separable_conv_layer(stage3_sconv8, 576, 3, 2, 'stage4_sconv1', activation=tf.nn.relu)
            stage4_sconv2 = self._separable_conv_layer(stage4_sconv1, 576, 3, 1, 'stage4_sconv2', activation=tf.nn.relu)
            stage4_sconv3 = self._separable_conv_layer(stage4_sconv2, 576, 3, 1, 'stage4_sconv3', activation=tf.nn.relu)
            stage4_sconv4 = self._separable_conv_layer(stage4_sconv3, 576, 3, 1, 'stage4_sconv4', activation=tf.nn.relu)

        downsampling_rate = 32
        return stage4_sconv4, downsampling_rate

    def _get_abbox(self, pshape, downsampling_rate):
        topleft_y = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        topleft_x = tf.range(0., tf.cast(pshape[2], tf.float32), dtype=tf.float32)
        topleft_y = tf.reshape(topleft_y, [-1, 1, 1, 1]) + 0.5
        topleft_x = tf.reshape(topleft_x, [1, -1, 1, 1]) + 0.5
        topleft_y = tf.tile(topleft_y, [1, pshape[2], 1, 1])
        topleft_x = tf.tile(topleft_x, [pshape[1], 1, 1, 1])
        topleft_yx = tf.concat([topleft_y, topleft_x], -1)
        topleft_yx = tf.tile(topleft_yx, [1, 1, self.num_anchors, 1])

        anchors = []
        for ratio in self.anchor_ratios:
            for size in self.anchor_scales:
                anchors.append([size*(ratio**0.5), size/(ratio**0.5)])
        anchors = tf.convert_to_tensor(anchors, tf.float32) / downsampling_rate
        anchors = tf.reshape(anchors, [1, 1, -1, 2])

        abbox_y1x1 = tf.reshape(topleft_yx - anchors / 2., [-1, 2])
        abbox_y2x2 = tf.reshape(topleft_yx + anchors / 2., [-1, 2])
        abbox_yx = abbox_y1x1 / 2. + abbox_y2x2 / 2.
        abbox_hw = abbox_y2x2 - abbox_y1x1
        return abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw

    def _one_image_rpn_train(self, pconf, pbbox_yx, pbbox_hw, abbox_yx,
                             abbox_hw, abbox_y1x1, abbox_y2x2, nground_truth, pshape):
        slice_index = tf.argmin(nground_truth, axis=0)[0]
        nground_truth = tf.gather(nground_truth, tf.range(0, slice_index, dtype=tf.int64))
        ngbbox_yx = nground_truth[..., 0:2]
        ngbbox_hw = nground_truth[..., 2:4]
        ngbbox_y1x1 = ngbbox_yx - ngbbox_hw / 2
        ngbbox_y2x2 = ngbbox_yx + ngbbox_hw / 2
        rcnn_label = tf.cast(nground_truth[..., 4], tf.int32)

        dpbbox_yx = pbbox_yx * abbox_hw + abbox_yx
        dpbbox_hw = abbox_hw * tf.exp(pbbox_hw)
        dpbbox_y1x1 = dpbbox_yx - dpbbox_hw / 2
        dpbbox_y2x2 = dpbbox_yx + dpbbox_hw / 2
        dpbbox_y1x1y2x2 = tf.concat([dpbbox_y1x1, dpbbox_y2x2], axis=-1)
        selected_indices = tf.image.non_max_suppression(
            dpbbox_y1x1y2x2, pconf[:, 0], self.post_nms_proposals, iou_threshold=0.5
        )
        pconf = tf.gather(pconf, selected_indices)
        pbbox_yx = tf.gather(pbbox_yx, selected_indices)
        pbbox_hw = tf.gather(pbbox_hw, selected_indices)
        abbox_yx = tf.gather(abbox_yx, selected_indices)
        abbox_hw = tf.gather(abbox_hw, selected_indices)
        abbox_y1x1 = tf.gather(abbox_y1x1, selected_indices)
        abbox_y2x2 = tf.gather(abbox_y2x2, selected_indices)
        proposal_yx = tf.gather(dpbbox_yx, selected_indices)
        proposal_hw = tf.gather(dpbbox_hw, selected_indices)

        num_ground_truth = tf.shape(ngbbox_yx)[0]
        num_abbox = tf.shape(abbox_yx)[0]

        ngbbox_y1x1ti = tf.reshape(ngbbox_y1x1, [-1, 1, 2])
        ngbbox_y2x2ti = tf.reshape(ngbbox_y2x2, [-1, 1, 2])
        ngbbox_y1x1ti = tf.tile(ngbbox_y1x1ti, [1, num_abbox, 1])
        ngbbox_y2x2ti = tf.tile(ngbbox_y2x2ti, [1, num_abbox, 1])
        abbox_y1x1ti = tf.reshape(abbox_y1x1, [1, -1, 2])
        abbox_y2x2ti = tf.reshape(abbox_y2x2, [1, -1, 2])
        abbox_y1x1ti = tf.tile(abbox_y1x1ti, [num_ground_truth, 1, 1])
        abbox_y2x2ti = tf.tile(abbox_y2x2ti, [num_ground_truth, 1, 1])

        gaiou_y1x1ti = tf.maximum(ngbbox_y1x1ti, abbox_y1x1ti)
        gaiou_y2x2ti = tf.minimum(ngbbox_y2x2ti, abbox_y2x2ti)
        gaiou_area = tf.reduce_prod(tf.maximum(gaiou_y2x2ti - gaiou_y1x1ti, 0), axis=-1)
        aarea = tf.reduce_prod(abbox_y2x2ti - abbox_y1x1ti, axis=-1)
        garea = tf.reduce_prod(ngbbox_y2x2ti - ngbbox_y1x1ti, axis=-1)
        gaiou_rate = gaiou_area / (aarea + garea - gaiou_area + 1e-8)
        best_raindex = tf.argmax(gaiou_rate, axis=1)

        best_pbbox_yx = tf.gather(pbbox_yx, best_raindex)
        best_pbbox_hw = tf.gather(pbbox_hw, best_raindex)
        best_pconf = tf.gather(pconf, best_raindex)
        best_abbox_yx = tf.gather(abbox_yx, best_raindex)
        best_abbox_hw = tf.gather(abbox_hw, best_raindex)
        best_proposal_yx = tf.gather(proposal_yx, best_raindex)
        best_proposal_hw = tf.gather(proposal_hw, best_raindex)
        best_rcnn_label = rcnn_label

        bestmask, _ = tf.unique(best_raindex)
        bestmask = tf.contrib.framework.sort(bestmask)
        bestmask = tf.reshape(bestmask, [-1, 1])
        bestmask = tf.sparse.SparseTensor(tf.concat([bestmask, tf.zeros_like(bestmask)], axis=-1),
                                          tf.squeeze(tf.ones_like(bestmask)), dense_shape=[num_abbox, 1])
        bestmask = tf.reshape(tf.cast(tf.sparse.to_dense(bestmask), tf.float32), [-1])
        othermask = (1. - bestmask) > 0.

        other_pbbox_yx = tf.boolean_mask(pbbox_yx, othermask)
        other_pbbox_hw = tf.boolean_mask(pbbox_hw, othermask)
        other_pconf = tf.boolean_mask(pconf, othermask)
        other_abbox_yx = tf.boolean_mask(abbox_yx, othermask)
        other_abbox_hw = tf.boolean_mask(abbox_hw, othermask)
        other_proposal_yx = tf.boolean_mask(proposal_yx, othermask)
        other_proposal_hw = tf.boolean_mask(proposal_hw, othermask)

        agiou_rate = tf.transpose(gaiou_rate)
        other_agiou_rate = tf.boolean_mask(agiou_rate, othermask)
        max_agiou_rate = tf.reduce_max(other_agiou_rate, axis=1)
        pos_mask = max_agiou_rate > 0.7
        neg_mask = max_agiou_rate < 0.3
        rgindex = tf.argmax(other_agiou_rate, axis=1)
        pos_rgindex = tf.boolean_mask(rgindex, pos_mask)
        pos_rcnn_label = tf.gather(rcnn_label, pos_rgindex)
        pos_ppox_yx = tf.boolean_mask(other_pbbox_yx, pos_mask)
        pos_ppox_hw = tf.boolean_mask(other_pbbox_hw, pos_mask)
        pos_pconf = tf.boolean_mask(other_pconf, pos_mask)
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_mask)
        pos_proposal_yx = tf.boolean_mask(other_proposal_yx, pos_mask)
        pos_proposal_hw = tf.boolean_mask(other_proposal_hw, pos_mask)
        pos_ngbbox_yx = tf.gather(ngbbox_yx, pos_rgindex)
        pos_ngbbox_hw = tf.gather(ngbbox_hw, pos_rgindex)
        neg_pconf = tf.boolean_mask(other_pconf, neg_mask)
        neg_proposal_yx = tf.boolean_mask(other_proposal_yx, neg_mask)
        neg_proposal_hw = tf.boolean_mask(other_proposal_hw, neg_mask)

        pos_rcnn_label = tf.concat([best_rcnn_label, pos_rcnn_label], axis=0)
        pos_pbbox_yx = tf.concat([best_pbbox_yx, pos_ppox_yx], axis=0)
        pos_pbbox_hw = tf.concat([best_pbbox_hw, pos_ppox_hw], axis=0)
        pos_pconf = tf.concat([best_pconf, pos_pconf], axis=0)
        pos_ngbbox_yx = tf.concat([ngbbox_yx, pos_ngbbox_yx], axis=0)
        pos_ngbbox_hw = tf.concat([ngbbox_hw, pos_ngbbox_hw], axis=0)
        pos_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        pos_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)
        pos_proposal_yx = tf.concat([best_proposal_yx, pos_proposal_yx], axis=0)
        pos_proposal_hw = tf.concat([best_proposal_hw, pos_proposal_hw], axis=0)

        num_pos = tf.shape(pos_pconf)[0]
        num_neg = tf.shape(neg_pconf)[0]
        chosen_num_neg = tf.cond(num_neg > 3*num_pos, lambda: 3*num_pos, lambda: num_neg)
        pos_rpn_label = tf.tile(tf.constant([0]), [num_pos])
        neg_rpn_label = tf.tile(tf.constant([1]), [num_neg])
        neg_rcnn_label = tf.tile(tf.constant([self.num_classes - 1]), [chosen_num_neg])

        pos_conf_loss = tf.losses.sparse_softmax_cross_entropy(labels=pos_rpn_label, logits=pos_pconf, reduction=tf.losses.Reduction.MEAN)
        neg_conf_loss = tf.losses.sparse_softmax_cross_entropy(labels=neg_rpn_label, logits=neg_pconf, reduction=tf.losses.Reduction.NONE)
        chosen_neg_conf_loss, chosen_neg_index = tf.nn.top_k(neg_conf_loss, chosen_num_neg)
        conf_loss = tf.reduce_mean(chosen_neg_conf_loss) + pos_conf_loss

        neg_proposal_yx = tf.gather(neg_proposal_yx, chosen_neg_index)
        neg_proposal_hw = tf.gather(neg_proposal_hw, chosen_neg_index)

        pos_truth_pbbox_yx = (pos_ngbbox_yx - pos_abbox_yx) / pos_abbox_hw
        pos_truth_pbbox_hw = tf.log(pos_ngbbox_hw / pos_abbox_hw)
        pos_yx_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_yx - pos_truth_pbbox_yx), axis=-1)
        pos_hw_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_hw - pos_truth_pbbox_hw), axis=-1)
        pos_coord_loss = tf.reduce_mean(pos_yx_loss + pos_hw_loss)

        total_loss = conf_loss + 10.0 * pos_coord_loss

        proposal_yx = tf.concat([pos_proposal_yx, neg_proposal_yx], axis=0)
        proposal_hw = tf.concat([pos_proposal_hw, neg_proposal_hw], axis=0)
        proposal_y1x1 = proposal_yx - proposal_hw / 2.
        proposal_y2x2 = proposal_yx + proposal_hw / 2.
        rcnn_label = tf.concat([pos_rcnn_label, neg_rcnn_label], axis=0)

        y_mask = tf.cast(proposal_yx[:, 0] > 0., tf.float32) * tf.cast(proposal_yx[:, 0] < tf.cast(pshape[1] - 1, tf.float32), tf.float32)
        x_mask = tf.cast(proposal_yx[:, 1] > 0., tf.float32) * tf.cast(proposal_yx[:, 1] < tf.cast(pshape[2] - 1, tf.float32), tf.float32)
        h_mask = tf.cast(proposal_hw[:, 0] > 0., tf.float32)
        w_mask = tf.cast(proposal_hw[:, 1] > 0., tf.float32)
        yxhw_mask = (y_mask * x_mask * h_mask * w_mask) > 0.
        proposal_y1x1 = tf.boolean_mask(proposal_y1x1, yxhw_mask)
        proposal_y2x2 = tf.boolean_mask(proposal_y2x2, yxhw_mask)
        rcnn_label = tf.boolean_mask(rcnn_label, yxhw_mask)
        pos_ngbbox_yx = tf.boolean_mask(pos_ngbbox_yx, tf.gather(yxhw_mask, tf.range(0, num_pos)))
        pos_ngbbox_hw = tf.boolean_mask(pos_ngbbox_hw, tf.gather(yxhw_mask, tf.range(0, num_pos)))
        y1 = proposal_y1x1[:, 0:1]
        x1 = proposal_y1x1[:, 1:2]
        y2 = proposal_y2x2[:, 0:1]
        x2 = proposal_y2x2[:, 1:2]
        y1 = tf.maximum(y1, tf.zeros_like(y1))
        x1 = tf.maximum(x1, tf.zeros_like(x1))
        y2 = tf.minimum(y2, tf.zeros_like(y2) + tf.cast(pshape[1] - 1, tf.float32))
        x2 = tf.minimum(x2, tf.zeros_like(x2) + tf.cast(pshape[2] - 1, tf.float32))
        proposal_y1x1 = tf.concat([y1, x1], axis=-1)
        proposal_y2x2 = tf.concat([y2, x2], axis=-1)
        proposal_yx = proposal_y2x2 / 2. + proposal_y1x1 / 2.
        proposal_hw = proposal_y2x2 - proposal_y1x1
        proposal = tf.concat([proposal_y1x1, proposal_y2x2], axis=-1)

        return total_loss, proposal, proposal_yx, proposal_hw, pos_ngbbox_yx, pos_ngbbox_hw, rcnn_label

    def _one_image_rpn_test(self, pconf, pbbox_yx, pbbox_hw, abbox_yx, abbox_hw, pshape):
        dpbbox_yx = pbbox_yx * abbox_hw + abbox_yx
        dpbbox_hw = abbox_hw * tf.exp(pbbox_hw)
        dpbbox_y1x1 = dpbbox_yx - dpbbox_hw / 2
        dpbbox_y2x2 = dpbbox_yx + dpbbox_hw / 2
        dpbbox_y1x1y2x2 = tf.concat([dpbbox_y1x1, dpbbox_y2x2], axis=-1)
        selected_indices = tf.image.non_max_suppression(
            dpbbox_y1x1y2x2, pconf[:, 0], self.post_nms_proposals, iou_threshold=0.5
        )
        pconf = tf.nn.softmax(tf.gather(pconf, selected_indices))
        proposal_yx = tf.gather(dpbbox_yx, selected_indices)
        proposal_hw = tf.gather(dpbbox_hw, selected_indices)

        pos_mask = pconf[:, 0] > 0.5
        pos_proposal_yx = tf.boolean_mask(proposal_yx, pos_mask)
        pos_proposal_hw = tf.boolean_mask(proposal_hw, pos_mask)

        y_mask = tf.cast(pos_proposal_yx[:, 0] > 0., tf.float32) * tf.cast(pos_proposal_yx[:, 0] < tf.cast(pshape[1] - 1, tf.float32), tf.float32)
        x_mask = tf.cast(pos_proposal_yx[:, 1] > 0., tf.float32) * tf.cast(pos_proposal_yx[:, 1] < tf.cast(pshape[2] - 1, tf.float32), tf.float32)
        h_mask = tf.cast(pos_proposal_hw[:, 0] > 0., tf.float32)
        w_mask = tf.cast(pos_proposal_hw[:, 1] > 0., tf.float32)
        yxhw_mask = (y_mask * x_mask * h_mask * w_mask) > 0.
        pos_proposal_yx = tf.boolean_mask(pos_proposal_yx, yxhw_mask)
        pos_proposal_hw = tf.boolean_mask(pos_proposal_hw, yxhw_mask)

        pos_proposal_y1x1 = pos_proposal_yx - pos_proposal_hw / 2.
        pos_proposal_y2x2 = pos_proposal_yx + pos_proposal_hw / 2.

        y1 = pos_proposal_y1x1[:, 0:1]
        x1 = pos_proposal_y1x1[:, 1:2]
        y2 = pos_proposal_y2x2[:, 0:1]
        x2 = pos_proposal_y2x2[:, 1:2]
        y1 = tf.maximum(y1, tf.zeros_like(y1))
        x1 = tf.maximum(x1, tf.zeros_like(x1))
        y2 = tf.minimum(y2, tf.zeros_like(y2) + tf.cast(pshape[1] - 1, tf.float32))
        x2 = tf.minimum(x2, tf.zeros_like(x2) + tf.cast(pshape[2] - 1, tf.float32))
        pos_proposal_y1x1 = tf.concat([y1, x1], axis=-1)
        pos_proposal_y2x2 = tf.concat([y2, x2], axis=-1)
        pos_proposal_yx = pos_proposal_y2x2 / 2. + pos_proposal_y1x1 / 2.
        pos_proposal_hw = pos_proposal_y2x2 - pos_proposal_y1x1
        pos_proposal = tf.concat([pos_proposal_y1x1, pos_proposal_y2x2], axis=-1)

        return pos_proposal_yx, pos_proposal_hw, pos_proposal

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'train':
            if self.train_initializer is not None:
                self.sess.run(self.train_initializer)

    def _create_pretraining_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_detection_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor')
        self.pretraining_weight_saver = tf.train.Saver(weights)
        weights = weights + tf.trainable_variables('regressor')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_pretraining_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def _create_detection_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _train_pretraining_epoch(self, lr):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        mean_acc = []
        for i in range(self.num_train // self.batch_size):
            _, loss, acc, summaries = self.sess.run([self.train_op, self.loss, self.accuracy, self.summary_op],
                                                    feed_dict={self.lr: lr})
            mean_loss.append(loss)
            mean_acc.append(acc)
        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        return mean_loss, mean_acc

    def _train_detection_epoch(self, lr):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        num_iters = self.num_train // self.batch_size
        for i in range(num_iters):
            _, loss, summaries = self.sess.run([self.train_op, self.loss, self.summary_op],
                                               feed_dict={self.lr: lr})
            sys.stdout.write('\r>> ' + 'iters '+str(i+1)+str('/')+str(num_iters)+' loss '+str(loss))
            sys.stdout.flush()
            mean_loss.append(loss)
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def _test_one_pretraining_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.pred, feed_dict={self.images: images})
        return pred

    def _test_one_detection_image(self, images):
        self.is_training = False
        pred = self.sess.run(self.detection_pred, feed_dict={self.images: images})
        return pred

    def _save_pretraining_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('>> save', mode, 'model in', path, 'successfully')

    def _save_detection_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('>> save', mode, 'model in', path, 'successfully')

    def load_weight(self, path):
        self.saver.restore(self.sess, path)
        print('>> load weight', path, 'successfully')

    def load_pretraining_weight(self, path):
        self.pretraining_weight_saver.restore(self.sess, path)
        print('>> load pretraining weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_layer(self, bottom, filters, kernel_size, strides, name=None, dilation_rate=1, activation=None):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _separable_conv_layer(self, bottom, filters, kernel_size, strides, name=None, dilation_rate=1, activation=None):
        conv = tf.layers.separable_conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )
