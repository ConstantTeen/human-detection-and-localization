#!/usr/bin/env python

"""
Helper functions and classes will be placed here.
"""

import os
import tarfile
import six.moves.urllib as urllib

import numpy as np

from cob_perception_msgs.msg import Detection, DetectionArray, Rect

from people_msgs.msg import DetectedPeople, Person, Rect as myRect


def create_people_detection_msg(im, output_dict, category_index, bridge):
    """
    Creates the body parts detection array message

    Args:
    im: (std_msgs_Image) incomming message

    output_dict (dictionary) output of object detection model

    category_index: dictionary of labels (like a lookup table)

    bridge (cv_bridge) : cv bridge object for converting

    Returns:

    msg (people_msgs/DetectedPeople) The message to be sent

    """

    boxes = output_dict["detection_boxes"]
    scores = output_dict["detection_scores"]
    classes = output_dict["detection_classes"]
    masks = None

    if 'detection_masks' in output_dict:
        masks = output_dict["detection_masks"]

    msg = DetectedPeople()

    msg.header = im.header

    scores_above_threshold = np.where(scores > 0.5)[0]
    
    parts = {
        52: [], # person
        177: [],# torso
        246: [] # face
    }
    
    for s in scores_above_threshold: 
        # Get the properties

        bb = boxes[s,:]
        sc = scores[s]
        cl = classes[s]
        
        x = int((im.width-1) * bb[1])
        y = int((im.height-1) * bb[0])
        width = int((im.width-1) * (bb[3]-bb[1]))
        height = int((im.height-1) * (bb[2]-bb[0]))
        
        if cl in parts.keys():
            parts[cl].append( (x,y,width,height) )
    
    detected_people = get_body_parts_set(parts)
    
    #create detection message
    for d in detected_people:
        detection = Person()
        
        (detection.person.x,detection.person.y,detection.person.h, detection.person.w) = d[0]
        (detection.face.x,detection.face.y,detection.face.h, detection.face.w) = d[1]
        (detection.torso.x,detection.torso.y,detection.torso.h, detection.torso.w) = d[2]
        
        msg.persons.append(detection)    

    return msg

    
def get_body_parts_set(parts):
    """
    Takes boxes of body parts and group them
    
    Args:
    parts: (dictionary) all faces, torsos and full bodies detected
    
    Returns:
    
    result: (python list) list of triplets of faces, torsos and full bodies corresponding to same peron
    """
    faces  = parts[246]
    torsos = parts[177]
    people = parts[52]
    
    f_len = len(faces)
    t_len = len(torsos)
    p_len = len(people)
    
    pf_intersection_matrix = np.matrix(
        [[None for i in range(f_len)] for j in range(p_len)]) 
    pt_intersection_matrix = np.matrix(
        [[None for i in range(t_len)] for j in range(p_len)])
    proximity_matrix = np.matrix(
        [[None for i in range(t_len)] for j in range(f_len)])
    
    #make intersections matrices
    for i in range(p_len):
        p = people[i]
        
        for j in range(f_len):
            f = faces[j]                
            pf_intersection_matrix[i,j] = get_area_percentage(p,f)
    
        for k in range(t_len):
            t = torsos[j]
            pt_intersection_matrix[i,j] = get_area_percentage(p,t)
            
    #make proximity matrix
    for i in range(f_len):
        f = faces[i]
        
        for j in range(t_len):
            t = torsos[j]
            proximity_matrix[i,j] = get_proximity_metric(t,f)
            
    
    result = []
    f_range = range(f_len)
    t_range = range(t_len)
    p_range = range(p_len)
    
    #form triplets
    if t_len*f_len*p_len > 0:
        for i in range(p_len):
            if i in p_range:
                max_face = np.argmax(pf_intersection_matrix[i,:])
                max_torso = np.argmax(pt_intersection_matrix[i,:])
                
                result.append((people[i],faces[max_face], torsos[max_torso]))
                
                f_range[max_face] = None
                t_range[max_torso] = None
                p_range[i] = None
    elif t_len*p_len > 0:
        for i in range(p_len):
            if i in p_range:
                max_torso = np.argmax(pt_intersection_matrix[i,:])
                
                result.append( (people[i],(-1,-1,-1,-1), torsos[max_torso]) )
                
                t_range[max_torso] = None
                p_range[i] = None
    elif f_len*p_len > 0:
        for i in range(p_len):
            if i in p_range:
                max_face = np.argmax(pf_intersection_matrix[i,:])
                
                result.append( (people[i],faces[max_face],(-1,-1,-1,-1)) )
                
                f_range[max_face] = None
                p_range[i] = None
    elif p_len > 0:
        for i in range(p_len):
            if i in p_range:
                result.append( 
                    (people[i],(-1,-1,-1,-1),(-1,-1,-1,-1)) )
                        
                p_range[i] = None
    elif t_len*f_len > 0:
        for i in range(t_len):
            if i in t_range:
                max_face = np.argmin(proximity_matrix[i,:])
                
                result.append( ((-1,-1,-1,-1), faces[max_face], torsos[i]) )
                
                t_range[i] = None
                f_range[max_face] = None
    elif f_len >0:
        for i in range(f_len):
            if i in f_range:
                result.append( 
                    ((-1,-1,-1,-1), faces[i], (-1,-1,-1,-1)) )
                
                f_range[i] = None
    elif t_len >0:
        for i in range(t_len):
            if i in t_range:
                result.append( 
                    ((-1,-1,-1,-1), (-1,-1,-1,-1), torsos[i]) )
                
                t_range[i] = None
    
    return result


def get_proximity_metric(rect1,rect2, k=0.5):
    """
    Calculates metric of rectangles matching. Two rectangles match if they are close enough and their centers lie on the same vertical (pretty much). 
    
    Args:
    rect1(/2): (tuple) contains x,y coordinates of leftmost rectangle vertex and its height and width
    
    k(float): parameter, lies in [0,1].
    
    Returns:
    metric(int): the much rectangles match the lower metric is
    """
    (x1,y1,w1,h1) = rect1
    (x2,y2,w2,h2) = rect2
    
    metric = k*abs(y1-y2-h2) + (1-k)*abs(x1+w1/2-x2-w2/2)
    
    return metric

def get_area_percentage(rect1,rect2):
    """
    Calculates metric of belonging rectangle rect1 to rectangle rect2
    
    Args:
    rect1(/2): (tuple) contains x,y coordinates of leftmost rectangle vertex and its height and width
    
    Returns:
    Quotient of rectangles intersection area and the second rectangle area
    """
    (x1,y1,w1,h1) = rect1
    (x2,y2,w2,h2) = rect2
    
    s1 = h1*w1 # first rectangle area
    s2 = h2*w2 # second rectangle area
    s12 = 0    # intersection area
    
    x = max(x1,x2)
    y = max(y1,y2)
    xx = min(x1+w1,x2+w2)
    yy = min(y1+h1,y2+h2)

    if(xx > x) and (yy > y):
        s12 = (xx-x)*(yy-y) 
    
    return float(s12)/s2



def download_model(\
    download_base='http://download.tensorflow.org/models/object_detection/', \
    model_name='ssd_mobilenet_v1_coco_11_06_2017'\
    ):
    """
    Downloads the detection model from tensorflow servers

    Args:
    download_base: base url where the object detection model is downloaded from

    model_name: name of the object detection model

    Returns:

    """

    # add tar gz to the end of file name
    model_file = model_name + '.tar.gz'

    try:
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, \
            model_file)
        tar_file = tarfile.open(model_file)
        for f in tar_file.getmembers():
            file_name = os.path.basename(f.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(f, os.getcwd())
    except Exception as e:
        raise

def create_detection_msg(im, output_dict, category_index, bridge):
    """
    Creates the detection array message

    Args:
    im: (std_msgs_Image) incomming message

    output_dict (dictionary) output of object detection model

    category_index: dictionary of labels (like a lookup table)

    bridge (cv_bridge) : cv bridge object for converting

    Returns:

    msg (cob_perception_msgs/DetectionArray) The message to be sent

    """

    boxes = output_dict["detection_boxes"]
    scores = output_dict["detection_scores"]
    classes = output_dict["detection_classes"]
    masks = None

    if 'detection_masks' in output_dict:
        masks = output_dict["detection_masks"]

    msg = DetectionArray()

    msg.header = im.header

    scores_above_threshold = np.where(scores > 0.5)[0]

    for s in scores_above_threshold:
        # Get the properties

        bb = boxes[s,:]
        sc = scores[s]
        cl = classes[s]
        print('box::::::::::::' + str(im.width) +'|'+ str(im.height))

        # Create the detection message
        detection = Detection()
        detection.header = im.header
        detection.label = category_index[int(cl)]['name']
        detection.id = cl
        detection.score = sc
        detection.detector = 'Tensorflow object detector'
        detection.mask.roi.x = int((im.width-1) * bb[1])
        detection.mask.roi.y = int((im.height-1) * bb[0])
        detection.mask.roi.width = int((im.width-1) * (bb[3]-bb[1]))
        detection.mask.roi.height = int((im.height-1) * (bb[2]-bb[0]))

        if 'detection_masks' in output_dict:
            detection.mask.mask = \
                bridge.cv2_to_imgmsg(masks[s], "mono8")

            print detection.mask.mask.width


        msg.detections.append(detection)

    return msg