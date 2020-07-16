''' Facial attribute extraction using mxnet and facenet '''
#--------------------------------
# Date : 10-07-2020
# Project : Facial Attribute Extraction
# Category : DeepLearning
# Company : weblineindia
# Department : AI/ML
#--------------------------------
import os
import cv2
import sys
import glob
import logging
import argparse
import numpy as np
import mxnet as mx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import model.emotion.detectemotion as ime
from mxnet_moon.lightened_moon import lightened_moon_feature

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# import pdb

# Load path from .env
faceProto = os.getenv("FACEDETECTOR")
faceModel = os.getenv("FACEMODEL")
ageProto = os.getenv("AGEDETECTOR")
ageModel = os.getenv("AGEMODEL")
genderProto = os.getenv("GENDERDETECTOR")
genderModel = os.getenv("GENDERMODEL")
pathImg = os.getenv("IMGPATH")
APPROOT = os.getenv("APPROOT")

#Load face detection model
faceNet=cv2.dnn.readNet(faceModel,faceProto)
#Load age detection model
ageNet=cv2.dnn.readNet(ageModel,ageProto)
#Load gender detection model
genderNet=cv2.dnn.readNet(genderModel,genderProto)
#create instance for emotion detection
ed = ime.Emotional()


""" Detects face and extracts the coordinates"""
def getFaceBox(net, image, conf_threshold=0.7):
    image=image.copy()
    imageHeight=image.shape[0]
    imageWidth=image.shape[1]
    blob=cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*imageWidth)
            y1=int(detections[0,0,i,4]*imageHeight)
            x2=int(detections[0,0,i,5]*imageWidth)
            y2=int(detections[0,0,i,6]*imageHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), int(round(imageHeight/150)), 8)
    return image,faceBoxes



""" Detects age and gender """
def genderAge(image,faceBox):

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']
    
    padding=20
    face=image[max(0,faceBox[1]-padding):
        min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
        :min(faceBox[2]+padding, image.shape[1]-1)]
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict the gender
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    # Predict the age
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    # Return
    return gender,age




""" Function for gender detection,age detection and """            
def main():
    symbol = lightened_moon_feature(num_classes=40, use_fuse=True)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    _, arg_params, aux_params = mx.model.load_checkpoint('model/lightened_moon/lightened_moon_fuse', 82)

    ''' Loading Image from directory and writing attributes into .txt file'''
    img_dir = os.path.join(pathImg)
    if os.path.exists(img_dir):
        names = os.listdir(pathImg)      
        img_paths = [name for name in names]
        for imge in range(4005,4005+len(names)):

            imge = "{:06d}.jpg".format(imge)
            path = pathImg+str(imge)
            print("Image Path",path)
            # read img and drat face rect
            image = cv2.imread(path)
            img = cv2.imread(path, -1)
            
            resultImg,faceBoxes=getFaceBox(faceNet,image)
            if not faceBoxes:
                print("No face detected")

            # Loop throuth the coordinates
            for faceBox in faceBoxes:               

                print("#====Detected Age and Gender====#")

                gender,age = genderAge(image,faceBox)
                print('Gender',gender)
                print('Age',age)

                # Predict emotions in the image
                print("#====Detected Emotion===========#")
                
                emlist = ed.emotionalDet(path,faceBox)
                print(emlist)
        
                # Detect the facial attributes using mxnet
                # crop face area
                left = faceBox[0]
                width = faceBox[2] - faceBox[0]
                top = faceBox[1]
                height =  faceBox[3] - faceBox[1]
                right = faceBox[2]
                bottom = faceBox[3]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pad = [0.25, 0.25, 0.25, 0.25] if args.pad is None else args.pad
                left = int(max(0, left - width*float(pad[0])))
                top = int(max(0, top - height*float(pad[1])))
                right = int(min(gray.shape[1], right + width*float(pad[2])))
                bottom = int(min(gray.shape[0], bottom + height*float(pad[3])))
                gray = gray[left:right, top:bottom]
                # resizing image and increasing the image size
                gray = cv2.resize(gray, (args.size, args.size))/255.0
                img = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
                # get image parameter from mxnet
                arg_params['data'] = mx.nd.array(img, devs)
                exector = symbol.bind(devs, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
                exector.forward(is_train=False)
                exector.outputs[0].wait_to_read()
                output = exector.outputs[0].asnumpy()
                # 40 facial attributes
                text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
                        "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
                        "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
                        "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
                        "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
                
                #Predict the results
                pred = np.ones(40)
                # create a list based on the attributes generated.
                attrDict = {}
                detectedAttributeList = []
                for i in range(40):
                    attr = text[i].rjust(20)
                    if output[0][i] < 0:
                        attrDict[attr] = 'No'
                    else:
                        attrDict[attr] = 'Yes'
                        detectedAttributeList.append(text[i])

                print("#====Detected Attributes========#")
                for attribute in detectedAttributeList:
                    print(attribute)                  
                # Write images into the results directory
                cv2.imwrite(APPROOT+'results/'+str(imge), resultImg) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict the face attribution of one input image")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--pad', type=float, nargs='+',
                                 help="pad (left,top,right,bottom) for face detection region")
    parser.add_argument('--model-load-prefix', dest = 'model_load_prefix', type=str, default='../model/lightened_moon/lightened_moon_fuse',
                        help='the prefix of the model to load')
    args = parser.parse_args()
       
    logging.info(args)
    main()
