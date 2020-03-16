import cv2
import os
import math
import numpy as np 

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_path_list(root_path):
    data = os.listdir(root_path)
    return data

    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

def get_class_names(root_path, train_names):
    img_path =[]
    img_class_id =[]
    for index, path in enumerate(train_names):
        folder_full_path = root_path+'/'+path
        img_list = os.listdir(folder_full_path)

        for photo in img_list:
            photo_full_path = folder_full_path + '/' + photo
            img_path.append(photo_full_path)
            img_class_id.append(index)
    return img_path,img_class_id

    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''

def get_train_images_data(image_path_list):
    imagesList = []
    for path in image_path_list:
        image = cv2.imread(path)
        imagesList.append(image)
    return imagesList

    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''

def detect_faces_and_filter(image_list, image_classes_list=None):
    cropped_img=[]
    face_location=[]
    cropped_id = []
    for i in range(len(image_list)):
        raw_img = image_list[i]
        gray_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
        #detect face

        detected_faces = cascade.detectMultiScale(gray_img, scaleFactor = 1.2,  minNeighbors =5)
        if(len(detected_faces)== 0 or len(detected_faces)>1):
            continue
        for face_rect in detected_faces:
            x,y,w,h = face_rect
            curr_face= gray_img[y:y+h,x:x+w]

            cropped_img.append(curr_face)
            face_location.append(face_rect)
            if image_classes_list:
                cropped_id.append(image_classes_list[i])
    return cropped_img,face_location,cropped_id

    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays,np.array(image_classes_list))
    return recognizer
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path, image_path_list):
    test_list = []
    for image in image_path_list:
        full_path = test_root_path+'/'+image
        test_img = cv2.imread(full_path)
        test_list.append(test_img)
    return test_list

    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''

def predict(classifier, test_faces_gray):
    result_list = []
    for test in test_faces_gray:
        result,_ = classifier.predict(test)
        result_list.append(result)
    return result_list
    
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    predictedList = []
    for result,image,rect in zip(predict_results,test_image_list,test_faces_rects):
        x,y,w,h = rect
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)

        text = train_names[result]
        cv2.putText(image,text,(x,y-1),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,255,0),1)
        predictedList.append(image)
    return predictedList

    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''

def combine_results(predicted_test_image_list):
    if len(predicted_test_image_list) > 1:
        result_array = predicted_test_image_list[0]
        data = len(predicted_test_image_list)
        for i in range(1,data):
            new_arr = predicted_test_image_list[i]
            result_array = np.hstack((result_array,new_arr))
        return result_array
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''

def show_result(image):
    img = image
    cv2.imshow('Result',img)
    cv2.waitKey(0)

    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)