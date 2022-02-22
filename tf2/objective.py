# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contrastive loss functions."""

from absl import flags

import tensorflow.compat.v2 as tf
# from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import tensorflow
from tensorflow.compat.v2.keras.losses import KLDivergence

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits):
  """Compute mean supervised loss over local batch."""
  losses = tf.keras.losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                  logits)
  return tf.reduce_mean(losses)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         strategy=None):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    strategy: context information for tpu.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if  strategy is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
    replica_context = tf.distribute.get_replica_context()
    replica_id = tf.cast(
        tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
  labels = tf.one_hot(tf.range(batch_size), batch_size)
  labels=labels*0.9 +(1-0.9)*(1-labels)
  labels=tf.concat([labels, labels-tf.linalg.diag(tf.linalg.diag_part(labels))],1)
  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  tf.print(logits_aa)
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
  if False:
      loss_fn = tf.nn.softmax_cross_entropy_with_logits
      loss_a = loss_fn(
          labels, tf.concat([logits_ab, logits_aa], 1))
      loss_b = loss_fn(
          labels, tf.concat([logits_ba, logits_bb], 1))
  else:
      loss_fn = KLDivergence(tf.keras.losses.Reduction.NONE)
      loss_a = loss_fn(
          labels, tf.concat([tf.nn.softmax(logits_ab), tf.nn.softmax(logits_aa)], 1))
      loss_b = loss_fn(
          labels, tf.concat([tf.nn.softmax(logits_ba), tf.nn.softmax(logits_bb)], 1))
  loss = tf.reduce_mean(loss_a + loss_b)

  return loss, logits_ab, labels

    
#TODO: precompute sims at start of run, also use tensor operations instead of scalat
def names2sims(names, embed_model, bsz, dataset='imagenet2012'):
    embeds = embed_model.lookup(names)   
    norm_embeds = tf.nn.l2_normalize(embeds,1)    
    sim_mat=tf.matmul(norm_embeds, norm_embeds, transpose_b=True)
    #sim_mat.set_shape([bsz,bsz])
    # def get_sims_outer(x):
    #     def get_sims_inner(y):
    #         return tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(ex,0),tf.nn.l2_normalize(ey,0)))
    #     return tf.map_fn(get_sims_inner,names, fn_output_signature=tf.float32)
    # sim_mat=tf.map_fn(get_sims_outer, names,fn_output_signature=tf.float32)
    return tf.math.square(sim_mat)

def ids2sims(ids, embed_model, bsz):
    embeds = embed_model.lookup(ids)   
    norm_embeds = tf.nn.l2_normalize(embeds,1)    
    sim_mat=tf.matmul(norm_embeds, norm_embeds, transpose_b=True)
    sim_mat.set_shape([None,None])
    # def get_sims_outer(x):
    #     def get_sims_inner(y):
    #         return tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(ex,0),tf.nn.l2_normalize(ey,0)))
    #     return tf.map_fn(get_sims_inner,names, fn_output_signature=tf.float32)
    # sim_mat=tf.map_fn(get_sims_outer, names,fn_output_signature=tf.float32)
    return tf.math.square(sim_mat)

def get_names(pred):
    label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    table=tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(label_dict.keys()), dtype=tf.int64), 
            tf.convert_to_tensor(list(label_dict.values()))), 
        default_value=tf.constant(''))
    return table.lookup(tf.argmax(pred))
def get_batch_sims(labels, embed_model, bsz, dataset='imagenet2012', method="simclr"):
    '''
    Args:
        labels: vector of one-hot labels with shape (bsz, num_classes).
    
    Returns:
        Similarity matrix of shape (bsz,bsz).
        
    '''
    ids = tf.argmax(labels,1)
    sims = ids2sims(ids, embed_model, bsz)
    #Get label names
    # if dataset=='imagenet2012':
    #     label_names = [i[0][1] for i in decode_predictions(labels, top=1)]
    # elif dataset=='cifar10':
    #     label_names= tf.map_fn(get_names, labels, fn_output_signature=tf.string)
    # sims=names2sims(label_names, embed_model, bsz, dataset)
    #Load CNNB similarity dict
    #sims = tf.matmul(labels,labels, transpose_b=True)#
    
    #sims = tf.convert_to_tensor(sims)
    return sims

def add_CNNB_loss(true_labels, 
                 hidden,
                 embed_model,
                         dataset='imagenet2012',
                         hidden_norm=True,
                         temperature=1.0,
                         strategy=None):
  """Compute loss for model.

  Args:
    true_labels: vector of labels.
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    strategy: context information for tpu.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]
  sims=get_batch_sims(true_labels, embed_model, batch_size, dataset)
  # Gather hidden1/hidden2 across replicas and create local labels.
  if False and strategy is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
    replica_context = tf.distribute.get_replica_context()
    reps = strategy.num_replicas_in_sync
    #sims.set_shape([512//reps, 512//reps])
    replica_id = tf.cast(
        tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels1=tf.concat([sims if i==replica_id else tf.zeros(sims.shape) for i in range(reps)],1)
    labels2=tf.concat([sims-tf.linalg.diag(tf.linalg.diag_part(sims)) if i==replica_id else tf.zeros(sims.shape) for i in range(reps)],1)
    labels=tf.concat([labels1,labels2],1)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
  else:
    #sims.set_shape([batch_size, batch_size])
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels=tf.concat([sims,sims-tf.linalg.diag(tf.linalg.diag_part(sims))],1)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

  #Calculate similarity between hidden representations from aug1 and from aug1
  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  # tf.print(true_labels)
  # tf.print(logits_aa)
  #Mask out entries corresponding to diagonal (self-similarity) so they are 0 once softmaxed
  logits_aa = logits_aa - masks * LARGE_NUM
  #Calculate similarity between hidden representations from aug2 and from aug2
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  #Mask out entries corresponding to diagonal (self-similarity) so they are 0 once softmaxed
  logits_bb = logits_bb - masks * LARGE_NUM
  #Calculate similarity between hidden representations from aug1 and from aug2
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  #Calculate similarity between hidden representations from aug2 and from aug1 
  #-> identical to above case if using single GPU
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
  #Calculate loss for aug1 samples by taking softmax over logits and then applying cross_entropy
  if False:
      loss_fn = tf.nn.softmax_cross_entropy_with_logits
      loss_a = loss_fn(
          #The identity part of labels (left-side) compares against sim(aug1,aug2); 
          #Zeros (right-side) compare against masked sim(aug1,aug1)
          labels, 
          #Horizontally concatenate sim(aug1, aug2) with sim(aug1,aug1)
          tf.concat([logits_ab, logits_aa], 1))
      #Take symmetrical loss for aug2 samples
      loss_b = loss_fn(
          labels, tf.concat([logits_ba, logits_bb], 1))
  else:
      loss_fn = KLDivergence(tf.keras.losses.Reduction.NONE)
      loss_a = loss_fn(
          labels, tf.concat([tf.nn.softmax(logits_ab), tf.nn.softmax(logits_aa)], 1))
      loss_b = loss_fn(
          labels, tf.concat([tf.nn.softmax(logits_ba), tf.nn.softmax(logits_bb)], 1))
  
  loss = tf.reduce_mean(loss_a + loss_b)

  return loss, logits_ab, labels


def tpu_cross_replica_concat(tensor, strategy=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if strategy is None or strategy.num_replicas_in_sync <= 1:
    return tensor

  num_replicas = strategy.num_replicas_in_sync

  replica_context = tf.distribute.get_replica_context()
  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[replica_context.replica_id_in_sync_group]],
        updates=[tensor],
        shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                            ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
