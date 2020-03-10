#!/usr/bin/env python
# coding: utf-8

import rospy
import numpy as np
#from object_and_scene_detection.msg import DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from people_msgs.msg import DetectedPeople

class ObjectVisualizer(object):
    def __init__(self):
        rospy.init_node('object_visualizer')
        
        ## possible types: arrows, cyliners, rects, boxes
        #self.visualization_types = rospy.get_param("~visualization_types",["arrows"])
        self.visualization_types = ['arrows']
        
        self.marker_pub = rospy.Publisher('detected_objects_markers', MarkerArray, queue_size = 1)
        
        
        rospy.Subscriber('/face_recognizer/faces/realworld/position', DetectedPeople, self.detected_objects_cb)
    
    
    def detected_objects_cb(self, msg):
        markers_msg = MarkerArray()
        cntr = 0
        for detected_person in msg.persons:
            body_parts_list = (detected_person.face, detected_person.torso, detected_person.person)
            
            for body_part in body_parts_list:
                if 'arrows' in self.visualization_types:
                    marker_msg = Marker()
                    
                    marker_msg.header.frame_id = msg.header.frame_id
                    marker_msg.header.stamp = rospy.Time.now()
                    marker_msg.id = cntr #??
                    cntr += 1
                                
                    marker_msg.action = Marker.ADD
                    marker_msg.type = Marker.ARROW
                    marker_msg.lifetime = rospy.Duration(0.5)
                    
                    marker_msg.pose.orientation.w = 1 #??
                    
                    start_point = Point()
                    marker_msg.points.append(start_point)
                    end_point = Point()
                    end_point.x = body_part.principal_vector.x
                    end_point.y = body_part.principal_vector.y
                    end_point.z = body_part.principal_vector.z
                    marker_msg.points.append(end_point)
                    
                    marker_msg.scale.x = 0.01
                    marker_msg.scale.y = 0.02
                    marker_msg.scale.z = 0.1
                    
                    marker_msg.color.r = 0
                    marker_msg.color.g = 1
                    marker_msg.color.b = 0
                    marker_msg.color.a = 1                        
                                                    
                    markers_msg.markers.append(marker_msg)
            
        self.marker_pub.publish(markers_msg)
                    
    def run(self):
        rospy.spin()
        
def main():
    ov = ObjectVisualizer()
    ov.run()

if __name__ == '__main__' :
    main()