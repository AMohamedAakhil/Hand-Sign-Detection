from keras import load_model

model = load_model(path) 
a = 27
def predict_image(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)

    result = gesture_names[np.argmax(pred_array)]
    
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score

camera = cv2.VideoCapture(0) #uses webcam for video

while camera.isOpened():
   
    
    k = cv2.waitKey(10)
    if k == 32: # if spacebar pressed
        frame = np.stack((frame,)*3, axis=-1)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.reshape(1, 224, 224, 3)
        prediction, score = predict_image(frame)
