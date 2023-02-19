from keras.layers import Input
from keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Activation, Dense,Dropout
from tensorflow.keras.layers import BatchNormalization

import cv2
import numpy as np
from statistics import mean,median
predictions = []
ensemble_predictions = []

def modelImg() :
    
    resnet50_input = Input(shape = (224, 224, 3), name = 'Image_input')

    ## The RESNET model
    model_resnet50_conv = ResNet50(weights= 'imagenet', include_top=False, input_shape= (224,224,3))



    output_resnet50_conv = model_resnet50_conv(resnet50_input)
    x=GlobalAveragePooling2D()(output_resnet50_conv)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.1)(x) # **reduce dropout 
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x=Dense(512,activation='relu')(x) #dense layer 3
    x = Dense(10, activation='softmax', name='predictions')(x)


    resnet50_pretrained = Model(inputs = resnet50_input, outputs = x)

    resnet50_pretrained.load_weights('resnet_weights_aug_extralayers_sgd_setval.hdf5')


    print('Model is loaded')






# Start capturing video from the camera
cap = cv2.VideoCapture(0)

# Loop over frames of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    resized = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(resized, axis=0)
    # labels is the image array


    model4_prediction = []
    model4_pred_class = []

    model4_prediction = resnet50_pretrained.predict(input_data)
    print(model4_prediction)
    mean_prediction = []
    for i in range(10):
        predictions.append(model4_prediction[0][i])
        trimmed_value = (sum(predictions) - max(predictions) - min(predictions))/(len(predictions) - 2)
        mean_value = mean(predictions)
        predictions = []
        mean_prediction.append(trimmed_value)
    mean_prediction = mean_prediction/ sum(mean_prediction)
    ensemble_predictions.append(mean_prediction)
    tags = { "C0": "safe driving","C1": "texting - right","C2": "On the phone - right","C3": "texting - left","C4": "On the phone - left",
"C5": "operating the radio","C6": "drinking","C7": "reaching behind","C8": "hair and makeup","C9": "talking to passenger" }

# labels is the image array

    i = 0
    tags_previous = ''
    cntr = 0
    #fig, ax = plt.subplots(20, 1, figsize = (200,200))



    for i in range(len(ensemble_predictions)):
    #for i in range(40,60):
        predicted_class = 'C'+str(np.where(ensemble_predictions[i] == np.amax(ensemble_predictions[i]))[0][0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = tags[predicted_class].upper()+':'+str(round(ensemble_predictions[i].max()*100,0))+'%'
        if tags[predicted_class] != tags_previous:
            cntr = 0

        if tags[predicted_class] == 'safe driving':
            print(text)
            cv2.rectangle(frame, (0, 0), (440, 100), (255, 255, 255), -1)
            cv2.putText(frame,text,(40,35), font, 0.8 ,(0,256,0),2)

        else:
            if cntr >= 0:
                print(text)
                cv2.rectangle(frame, (0, 0), (440, 100), (255, 255, 255), -1)
                cv2.putText(frame,text,(40,35), font, 0.8 ,(0,0,256),2)
            else:
                text = str(cntr+1)
                cntr = cntr+1
                print('came here')
            print(text)
            cv2.rectangle(frame, (0, 0), (440, 100), (255, 255, 255), -1)
            cv2.putText(frame,text,(40,35), font, 0.8 ,(0,0,256),2)

        tags_previous = tags[predicted_class]

    

    


    

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
