# Code for implementation of forward pass of retrained MobileNet
#Borrowed heavily from 
#Tensorflow transfer learning image classification tutorial: https://www.tensorflow.org/tutorials/image_retraining
#Multi thread for faster processing borrowed from : Dat Tran and his blog article :https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
# The internet in general. Thanks dudes.
#TODO: Comment this code


import argparse

import numpy as np
import tensorflow as tf
import cv2
import multiprocessing
from multiprocessing import Queue, Pool
import time
import struct
import six
import collections
import datetime
from threading import Thread
from matplotlib import colors




class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()



def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def read_tensor_from_jpg_image(input_image,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):


  float_caster = tf.cast(input_image, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def worker(input_q, output_q,input_model_path,input_layer_path,output_layer_path,img_mean,img_std,img_height,img_width,label_file_path):
    # Load a (frozen) Tensorflow model into memory.
    model_file = input_model_path
    input_layer = input_layer_path
    output_layer = output_layer_path
    input_height = img_height
    input_width = img_width
    input_mean = img_mean
    input_std = img_std
    label_file = label_file_path

    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        fps = FPS().start()
        while True:
            fps.update()
            frame = input_q.get()
            t = read_tensor_from_jpg_image(
                frame,
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std)

            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
            results = np.squeeze(results)
            labels = load_labels(label_file)
            prediction_string = str(labels[0]) + ' ' + ':' + ' ' + str(results[0])
            cv2.putText(img=frame, text=prediction_string, org=(int(input_width / 2), int(input_height / 2)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5,
                        color=(0, 255, 0))
            output_q.put(frame)

        fps.stop()
        sess.close()





if __name__ == '__main__':

    model_file = r'C:\tmp\output_graph.pb'
    label_file = r'C:\tmp\output_labels.txt'
    vid_path = r'C:\Users\PaballoM\OneDrive\ML Projects\Deep Learning\Applications\Ad_Dectection\Hollard\creative\Hollard_Funeral_Plan.mp4'
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    input_layer = 'module_apply_default/hub_input/Mul'
    output_layer = 'final_result'
    queue_size = 50
    num_workers = 3


    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    pool = Pool(num_workers, worker, (input_q, output_q,model_file,input_layer,output_layer,input_mean,input_std,input_height,input_width,label_file))
    cap = cv2.VideoCapture(vid_path)
    fps = FPS().start()

   # i = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        input_q.put(frame)

        t = time.time()
        output_frame = output_q.get()
        cv2.imshow('frame', output_frame)
       # cv2.imwrite(r'C:\Users\PaballoM\OneDrive\ML Projects\Deep Learning\Applications\Ad_Dectection\Hollard\detection_result\predict_res_{}'.format(i))
        #i = i+1
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    cv2.destroyAllWindows()

