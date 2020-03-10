#!/usr/bin/env python
"""
A ROS node to get 3D values of bounding boxes returned by face_recognizer node.

This node gets the face bounding boxes and gets the real world coordinates of
them by using depth values. It simply gets the x and y values of center point
and gets the median value of face depth values as z value of face.

"""

import rospy

import message_filters

import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo

#from cob_perception_msgs.msg import DetectionArray
from people_msgs.msg import DetectedPeople

class ProjectionNode(object):
    """Get 3D values of bounding boxes returned by face_recognizer node.

    _bridge (CvBridge): Bridge between ROS and CV image
    pub (Publisher): Publisher object for face depth results
    fx (Float): X component of Focal Length
    fy (Float): Y component of Focal Length
    cx (Int): Principle Point Horizontal
    cy (Int): Principle Point Vertical

    """
    def __init__(self):
        super(ProjectionNode, self).__init__()

        # init the node
        rospy.init_node('my_projection_node', anonymous=False)

        self._bridge = CvBridge()

        (depth_topic, people_topic, camera_topic, output_topic) = \
            self.get_parameters()

         # Subscribe to the face positions
        sub_obj = message_filters.Subscriber(people_topic,\
            DetectedPeople)

        sub_depth = message_filters.Subscriber(depth_topic,\
            Image)
        
        sub_info = rospy.Subscriber(camera_topic,\
            CameraInfo, self.get_camera_params)

        # Advertise the result of People Depths
        self.pub = rospy.Publisher(output_topic, \
            DetectedPeople, queue_size=1)

        # Create the message filter
        ts = message_filters.ApproximateTimeSynchronizer(\
            [sub_obj, sub_depth], \
            2, \
            1.4) #0.9

        ts.registerCallback(self.detection_callback)

        # spin
        rospy.spin()


    def get_camera_params(self, msg):
        """
        Callback for Camera Info.
        
        Args:
        msg (sensor_msgs/CameraInfo): camera information. Check the official documentation for details
        """
        (self.fx, self.fy, self.cx, self.cy) = (msg.K[0],msg.K[4],msg.K[2],msg.K[5])
        
        

    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def detection_callback(self, msg, depth):
        """
        Callback for RGB images: The main logic is applied here

        Args:
        msg (people_msgs/DetectedPeople): detections array
        depth (sensor_msgs/PointCloud2): depth image from camera

        """
        
        cv_depth = self._bridge.imgmsg_to_cv2(depth, "passthrough")

        # get the number of detections
        no_of_detections = len(msg.persons)

        # Check if there is a detection
        if no_of_detections > 0:
            
            for i, detection in enumerate(msg.persons):
                (face, torso, person) = (detection.face, detection.torso, detection.person)
                
                for j, rect in enumerate((face, torso, person)):
                    
                    x = rect.x
                    y = rect.y
                    width = rect.w
                    height = rect.h

                    cv_depth_bounding_box = cv_depth[y:y+height,x:x+width]

                    try:

                        depth_mean = np.nanmedian(\
                        cv_depth_bounding_box[np.nonzero(cv_depth_bounding_box)])

                        real_x = (x + width/2-self.cx)*(depth_mean*0.001)/self.fx

                        real_y = (y + height/2-self.cy)*(depth_mean*0.001)/self.fy
                        
                        real_z = depth_mean*0.001
                        
                        if j == 0:
                            msg.persons[i].face.principal_vector.x = real_x
                            msg.persons[i].face.principal_vector.y = real_y
                            msg.persons[i].face.principal_vector.z = real_z
                        elif j == 1:
                            msg.persons[i].torso.principal_vector.x = real_x
                            msg.persons[i].torso.principal_vector.y = real_y
                            msg.persons[i].torso.principal_vector.z = real_z
                        elif j == 2:
                            msg.persons[i].person.principal_vector.x = real_x
                            msg.persons[i].person.principal_vector.y = real_y
                            msg.persons[i].person.principal_vector.z = real_z
                        else:
                            raise

                    except Exception as e:
                        print e
                        
        self.pub.publish(msg)

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Returns:
        (tuple) :
            depth_topic (String): Incoming depth topic name
            people_topic (String): Incoming people bounding boxes set topic name
            output_topic (String): Outgoing depth topic name
            camera_topic (String): Outgoing camera topic name
        """

        depth_topic  = rospy.get_param("~depth_topic")
        people_topic = rospy.get_param('~people_topic')
        output_topic = rospy.get_param('~output_topic')
        camera_topic = rospy.get_param('~camera_topic')

        return (depth_topic, people_topic, camera_topic, output_topic)


def main():
    """ main function
    """
    node = ProjectionNode()

if __name__ == '__main__':
    main()
