import rospy
import cv2
import mxnet as mx
import argparse
import threading
import time
from rcnn.tools.wrapper import MaskRCNN
from sensor_msgs.msg import Image
from threading import Lock
from cv_bridge import CvBridge, CvBridgeError

class SegmentationServer:
  """ This class loads a trained NN model that produces 
     segmentation and bbox results for given image topic.
     The segmented result is published to a channel.
  """
  def __init__(self, nn_model_file):
    self.rcnn_model = MaskRCNN(network = 'resnet_fpn', 
    						   prefix = nn_model_file,
    						   datatype = 'Blender_b',
    						   epoch = 0, 
    						   ctx = mx.gpu(0),
                   thresh = 0.5)
    self.image_lock = Lock()
    self.compute_lock = Lock()
    self.bridge = CvBridge()
    self.flag_compute = False
    self.image_pub_bbox = rospy.Publisher("det_img", Image)
    self.image_pub_mask = rospy.Publisher("mask_img", Image)
    self.image_pub_echo = rospy.Publisher("prior_img", Image)
    self.flag_received_image = False

  def subscribe_image_topic(self, topic_name):
  	self.rgb_sub = rospy.Subscriber(topic_name, Image, 
      self.rgb_update_cb, queue_size = 1)

  def rgb_update_cb(self, data):
    self.image_lock.acquire()
    try:
      self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      #print("got new image")
      self.flag_received_image = True
    except CvBridgeError as e:
      print(e)
    finally:
      #print("done with new image")
      self.image_lock.release()

    # Perform nn segmentation and publish the result.
    # self.compute_lock.acquire()
    # try:
    #   if self.flag_compute:
    #     #self.image_pub_echo.publish(self.bridge.cv2_to_imgmsg(self.cv_rgb_image, "bgr8"))
    #     #boxes, masks = self.rcnn_model.predict(self.cv_rgb_image, render=False)
    #     #det_map, mask_map = self.rcnn_model.visualize(self.cv_rgb_image, boxes, masks) 
    #     #self.image_pub.publish(self.bridge.cv2_to_imgmsg(det_map, "bgr8"))  
    # except:
    #   print("Segmentation prediction error")
    # finally:
    #   self.compute_lock.release()  

  def do_segmentation(self):
    while not rospy.is_shutdown():
      #print("do work")
      #self.image_lock.acquire()
      #print("locking")
      
      if self.flag_received_image:
        #print "do real work"
        self.image_lock.acquire()
        try:
          self.proc_image = self.cv_rgb_image.copy()
        except:
          print("error making a copy for nn prediction")
        finally:
          self.image_lock.release()    
          boxes, masks = self.rcnn_model.predict(self.proc_image, render=False)
          det_map, mask_map = self.rcnn_model.visualize(self.proc_image, boxes, masks) 
          self.image_pub_bbox.publish(self.bridge.cv2_to_imgmsg(det_map, "bgr8"))  
          self.image_pub_mask.publish(self.bridge.cv2_to_imgmsg(mask_map))
          cv2.imwrite('det.png', det_map)
          cv2.imwrite('seg.png', mask_map)
          #print("release lock")
      print("done work")

  def start_segmentation(self):
    if seg_server.flag_received_image:
      t = threading.Thread(name='do_segmentation', target=seg_server.do_segmentation)
      t.start()
    # self.compute_lock.acquire()
    # self.flag_compute = True 
    # self.compute_lock.release()

if __name__ == '__main__':
  rospy.init_node('segmentation_server', anonymous=True)
  parser = argparse.ArgumentParser(description='Test Mask-RCNN network')
  parser.add_argument('--topic', help='subscribed topic name', type=str)
  parser.add_argument('--model', help='pretrained model', type=str)
  args = parser.parse_args()  
  seg_server = SegmentationServer(nn_model_file = args.model)  
  seg_server.subscribe_image_topic(topic_name = args.topic)
  time.sleep(2)
  
  seg_server.start_segmentation()
  rospy.spin()