#!/usr/bin/env python

import rospy

import cv2

from cv_bridge import CvBridge, CvBridgeError

from cob_people_object_detection_tensorflow import utils

from people_msgs.msg import DetectedPeople

import csv

import numpy as np

import datetime


class DataCollector(object):
    def __init__(self):
        super(DataCollector, self).__init__()
        # init the node
        rospy.init_node('data_collection', anonymous=False)
        
        self._bridge = CvBridge()
        
        (depth_topic, self.output_folder) = self.get_parameters()
        
        sub_kinect = rospy.Subscriber(depth_topic,\
            DetectedPeople, self.callback)
        
        # set output file name
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.output_file_path = self.output_folder + now
        
        # spin
        rospy.spin()
        
    
    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Returns:
        (tuple) :
            depth_topic (String): Incoming set of grouped rectangles
        """

        depth_topic = rospy.get_param("~depth_topic")
        output_folder = rospy.get_param("~output_folder_name")
        

        return (depth_topic, output_folder)
        
        
    def callback(self, msg):
        
        detections = msg.persons
        
        with open(self.output_file_path, 'a') as csvfile:
            #fieldnames = ['face_x', 'face_y','face_width','face_height','face_cam_x', 'face_cam_y','face_cam_z',
                          #'torso_x','torso_y','torso_width','torso_height','torso_cam_x','torso_cam_y','torso_cam_z',
                          #'person_x','person_y','person_width','person_height','person_cam_x','person_cam_y','person_cam_z','mean_cam_z']
            writer = csv.writer(csvfile, delimiter=',')
            
            for d in detections:
                
                (face, torso, person) = (d.face, d.torso, d.person)
                
                mean = np.median([elem for elem in [face.principal_vector.z,torso.principal_vector.z,person.principal_vector.z] if (not np.isnan(elem)) and (elem > 0) ])
                
                values = [face.x,  face.y,  face.w,  face.h,  face.principal_vector.x,  face.principal_vector.y,  face.principal_vector.z,
                                 torso.x, torso.y, torso.w, torso.h, torso.principal_vector.x, torso.principal_vector.y, torso.principal_vector.z,
                                 person.x,person.y,person.w,person.h,person.principal_vector.x,person.principal_vector.y,person.principal_vector.z, mean]
                print(values)
                writer.writerow(values)
        
        
def main():
    node = DataCollector()
    
if __name__ == '__main__':
    main()

    
    