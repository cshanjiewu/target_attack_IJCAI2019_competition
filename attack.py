import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from perlin import create_perlin_noise
from nets import resnet_v1, inception, vgg, inception_resnet_v2, inception_v3, resnet_v2
slim = tf.contrib.slim

# parameter of model
CHECKPOINTS_DIR = '../target-attack-hanjie/model/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg16', 'vgg_16.ckpt'),
    'inception_resnet_v2': os.path.join(CHECKPOINTS_DIR, 'inception_resnet_v2_base', 'model.ckpt'),
    'inception_v3': os.path.join(CHECKPOINTS_DIR, 'inception_v3_base', 'model.ckpt'),
    'inception_v4': os.path.join(CHECKPOINTS_DIR, 'inception_v4_base', 'model.ckpt'),
    'resnet_v2_152': os.path.join(CHECKPOINTS_DIR, 'resnet_v2_152_base', 'model.ckpt'),
}

tf.flags.DEFINE_string(
    'input_dir', './img/', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', './output/', 'Output file to save labels.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size',11, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')

FLAGS = tf.flags.FLAGS

max_epsilon = 25.0
num_iter = 32
momentum = 1

# dropout operation
def precalc_jitter_mask(width=5):
    # Prepare a jitter mask with XOR (alternating).
    jitter_width = width
    jitter_mask = np.empty((FLAGS.batch_size, FLAGS.image_height, FLAGS.image_height, 3), dtype=np.bool)
    for i in range(FLAGS.image_height):
        for j in range(FLAGS.image_height):
            jitter_mask[:, i, j, :] = (i % jitter_width == 0) ^ (j % jitter_width == 0)
    return tf.convert_to_tensor(jitter_mask, dtype = tf.bool)


def generate_jitter_sample(X_orig, X_aex, fade_eps=0.1, width = 11):
    jitter_mask = precalc_jitter_mask(width=width)

    jitter_mask = tf.cast(jitter_mask, dtype=tf.float32)

    jitter_diff = (X_aex - X_orig) * jitter_mask

    X_candidate = X_aex - fade_eps * jitter_diff
    return X_candidate


# preprocess
def preprocess_for_model(images, model_type):
    if 'inception_resnet_v2' in model_type.lower() or 'inception_v3' in model_type.lower() or 'inception_v4' in model_type.lower():
        images = tf.image.resize_bilinear(images, [299, 299], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'inception_v1' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images


def load_images_with_target_label(input_dir):
    images = []
    filenames = []
    target_labels = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['targetedLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        images.append(image)
        filenames.append(filename)
        target_labels.append(filename2label[filename])
        idx += 1
        if idx == FLAGS.batch_size:
            images = np.array(images)
            yield filenames, images, target_labels
            filenames = []
            images = []
            target_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, target_labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with open(os.path.join(output_dir, filename), 'wb') as f:
            image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            # resize back to [299, 299]
            image = imresize(image, [299, 299])
            Image.fromarray(image).save(f, format='PNG')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def eval(adv_imgs, labels, x_inputs, total_score, total_count):
    image = (((adv_imgs + 1.0) * 0.5) * 255.0)
    processed_imgs_inv1 = preprocess_for_model(image, 'inception_v1')
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            processed_imgs_inv1, num_classes=FLAGS.num_classes, is_training=False, scope='InceptionV1', reuse=True)
    pred_inception = tf.argmax(end_points_inc_v1['Predictions'], 1)

    # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
    processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
            processed_imgs_res_v1_50, num_classes=FLAGS.num_classes, is_training=False, scope='resnet_v1_50', reuse=True)

    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])
    pred_resnet = tf.argmax(end_points_res_v1_50['probs'], 1)

    processed_imgs_inv3 = preprocess_for_model(image, 'inception_v3')
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_res_inception_v3, end_points_inception_v3 = inception_v3.inception_v3(
            processed_imgs_inv3, num_classes=FLAGS.num_classes, is_training=False, scope='InceptionV3', reuse=True)
    pred_inception_v3 = tf.argmax(end_points_inception_v3['Predictions'], 1)

    processed_imgs_inv_res = preprocess_for_model(image, 'inception_resnet_v2')
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_inception_resnet, end_points_inception_resnet = inception_resnet_v2.inception_resnet_v2(
            processed_imgs_inv_res, num_classes=FLAGS.num_classes, is_training=False, scope='InceptionResnetV2')
    pred_ince_res = tf.argmax(end_points_inception_resnet['Predictions'], 1)


    for i in range(adv_imgs.shape[0]):
        def f1(total_score, total_count):
            total_score = tf.add(total_score, 64)
            return total_score, total_count

        def f2(total_score, total_count):
            adv = (((adv_imgs[i] + 1.0) * 0.5) * 255.0)
            ori = (((x_inputs[i] + 1.0) * 0.5) * 255.0)
            diff = tf.reshape(adv, [-1, 3]) - tf.reshape(ori, [-1, 3])
            distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1)))
            total_score = tf.add(total_score, distance)
            total_count = tf.add(total_count, 1)
            return total_score, total_count
        total_score, total_count = tf.cond(tf.equal(pred_inception[i], labels[i]), lambda: f2(total_score, total_count), lambda: f1(total_score, total_count))
        total_score, total_count = tf.cond(tf.equal(pred_resnet[i], labels[i]), lambda: f2(total_score, total_count), lambda: f1(total_score, total_count))
        # total_score, total_count = tf.cond(tf.equal(pred_inception_v3[i], labels[i]), lambda: f2(total_score, total_count), lambda: f1(total_score, total_count))
        total_score, total_count = tf.cond(tf.equal(pred_ince_res[i], labels[i]), lambda: f2(total_score, total_count), lambda: f1(total_score, total_count))

    return total_score, total_count

# input diversity
def structure(input_tensor):
    """
    Args:
        input_tensor: NHWC
    """
    rnd = tf.random_uniform((), 210, 224, dtype=tf.int32)
    rescaled = tf.image.resize_images(
        input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = 224 - rnd
    w_rem = 224 - rnd
    # 0 is better
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                        pad_left, pad_right], [0, 0]])
    padded.set_shape((input_tensor.shape[0], 224, 224, 3))
    output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                         lambda: padded, lambda: input_tensor)
    return output


# input diversity
def structure_res(input_tensor):
    rnd = tf.random_uniform((), 269, 299, dtype=tf.int32)
    rescaled = tf.image.resize_images(
        input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = 299 - rnd
    w_rem = 299 - rnd
    # 0 is better
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                        pad_left, pad_right], [0, 0]])
    padded.set_shape((input_tensor.shape[0], 299, 299, 3))
    output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                         lambda: padded, lambda: input_tensor)
    return output


def target_graph(x, y, i, x_max, x_min, grad, label, x_ori):
    eps = 2.0 * max_epsilon / 255.0
    alpha = eps / num_iter
    num_classes = FLAGS.num_classes

    # find adversarial example in perlin noise direction
    x = x + tf.random_uniform(tf.shape(x), minval=-1e-2, maxval=1e-2) * \
        create_perlin_noise(seed=None, color=True, batch_size=FLAGS.batch_size, image_size=FLAGS.image_height, normalize=True, precalc_fade=None)

    # dropout
    x = generate_jitter_sample(x_ori, x, fade_eps=0, width=np.random.randint(low=1, high=12))

    image = (((x + 1.0) * 0.5) * 255.0)
    processed_imgs_in_v1 = preprocess_for_model(image, 'inception_v1')
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            structure(processed_imgs_in_v1), num_classes=num_classes, is_training=False, scope='InceptionV1')

    # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
    processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
            structure(processed_imgs_res_v1_50), num_classes=num_classes, is_training=False, scope='resnet_v1_50')

    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['while/resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

    # image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
    processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            structure(processed_imgs_vgg_16), num_classes=num_classes, is_training=False, scope='vgg_16')

    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

    processed_imgs_in_v3 = preprocess_for_model(image, 'inception_v3')
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_in_v3, end_points_in_v3 = inception_v3.inception_v3(
            structure_res(processed_imgs_in_v3), num_classes=num_classes, is_training=False, scope='InceptionV3')

    processed_res_v2 = preprocess_for_model(image, 'resnet_v2_152')
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_res_v2, end_points_res_v2 = resnet_v2.resnet_v2_152(
            structure_res(processed_res_v2), num_classes=num_classes, is_training=False, scope='resnet_v2_152')
    end_points_res_v2['logits'] = tf.squeeze(end_points_res_v2['resnet_v2_152/logits'], [1, 2])
    end_points_res_v2['probs'] = tf.nn.softmax(end_points_res_v2['logits'])

    one_hot = tf.one_hot(y, num_classes)

    # separate the loss
    cross_entropy_v1 = tf.losses.softmax_cross_entropy(one_hot,
                                                    end_points_inc_v1['Logits'],
                                                    label_smoothing=0.0,
                                                    weights=1.0)

    cross_entropy_re = tf.losses.softmax_cross_entropy(one_hot,
                                                    end_points_res_v1_50['logits'],
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy_vgg = tf.losses.softmax_cross_entropy(one_hot,
                                                    end_points_vgg_16['logits'],
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy_re2 = tf.losses.softmax_cross_entropy(one_hot,
                                                    end_points_res_v2['logits'],
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy_v3 = tf.losses.softmax_cross_entropy(one_hot,
                                                     end_points_in_v3['Logits'],
                                                     label_smoothing=0.0,
                                                     weights=1.0)


    pred = tf.argmax(end_points_inc_v1['Predictions'] + end_points_res_v1_50['probs'] + end_points_vgg_16['probs']+end_points_res_v2['probs'], 1)


    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    label = first_round * pred + (1 - first_round) * label

    noise_re = tf.gradients(cross_entropy_re,x)[0]
    noise_re2 = tf.gradients(cross_entropy_re2,x)[0]
    noise_v1 = tf.gradients(cross_entropy_v1,x)[0]
    noise_vgg = tf.gradients(cross_entropy_vgg,x)[0]
    noise_v3 = tf.gradients(cross_entropy_v3,x)[0]

    noise_re = tf.Print(noise_re, [i, cross_entropy_re, cross_entropy_re2, cross_entropy_v1, cross_entropy_vgg, cross_entropy_v3])

    noise_re = noise_re / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise_re, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])
    noise_re2 = noise_re2 / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise_re2, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])
    noise_v1 = noise_v1 / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise_v1, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])
    noise_vgg = noise_vgg / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise_vgg, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])
    noise_v3 = noise_v3 / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise_v3, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])

    noise = momentum * grad + noise_re + noise_re2 + noise_v1 + noise_vgg + noise_v3

    noise = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])

    x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
    x = tf.clip_by_value(x, x_min, x_max)

    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise, label, x_ori


def stop(x, y, i, x_max, x_min, grad, label, x_ori):
    return tf.less(i, num_iter)


def refine_advimages(ori, adv, pre_labels, target_lists):
    for i in range(len(target_lists)):
        if pre_labels[i] == target_lists[i]:
            adv[i, :, :, :] = ori[i, :, :, :]
    return adv


# Momentum Iterative FGSM
def main(_):
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_file
    # some parameter
    eps = 2.0 * max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    check_or_create_dir(output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])

        # preprocessing for model input,
        # note that images for all classifier will be normalized to be in [-1, 1]
        processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_ori = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.placeholder(tf.int64, shape=[FLAGS.batch_size])
        label = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, pre_label, _ = tf.while_loop(stop, target_graph, [x_input, y, i, x_max, x_min, grad, label,
                                                                                x_ori])

        # eval
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size])
        batch_score = tf.constant(0.0)
        batch_count = tf.constant(0.0)
        score, count = eval(adv_input, labels, x_input, batch_score, batch_count)
        total_score = 0.0
        total_count = 0

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])
            s4.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s5.restore(sess, model_checkpoint_map['inception_v3'])
            s6.restore(sess, model_checkpoint_map['resnet_v2_152'])
            start_time = time.time()

            for filenames, raw_images, target_labels in load_images_with_target_label(input_dir):
                # preprocess images
                processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                # generate adversarial images
                adv_images, pre_labels = sess.run([x_adv,pre_label], feed_dict={x_input: processed_imgs_, y: target_labels,
                                                                                 x_ori: processed_imgs_})
                # turn to original image if target label is the true label
                adv_images = refine_advimages(processed_imgs_, adv_images, pre_labels, target_labels)
                save_images(adv_images, filenames, output_dir)
                sub_score, sub_count = sess.run([score,count], feed_dict={adv_input: adv_images, labels: target_labels, x_input: processed_imgs_})
                total_score = total_score + sub_score
                total_count = total_count + sub_count
        elapsed_time = time.time() - start_time

    print('Finish! Time:{}'.format(elapsed_time))
    print('total score: {} total count: {}'.format(total_score/110/3, total_count/3))

if __name__ == '__main__':
    tf.app.run()