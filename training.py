from gc import callbacks
import tensorboard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf 


tensor_board= TensorBoard(log_dir='My_model')

model=tf.keras.Sequential([
    	tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
    	tf.keras.layers.MaxPooling2D(2,2),

    	tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    	tf.keras.layers.MaxPooling2D(3,3),

    	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    	tf.keras.layers.MaxPooling2D(2,2),

	    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    	tf.keras.layers.MaxPooling2D(2,2),

    	tf.keras.layers.Flatten(),
    	tf.keras.layers.Dense(128, activation='relu'),
    	tf.keras.layers.Dense(27, activation='softmax')]
)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
train_datagen = ImageDataGenerator(rescale=1./250)
training_set = train_datagen.flow_from_directory(r'C:\PROJECTS\SIGN_LANGUAGE_2\data\train',
                                                 target_size=(64,64),
                                                 batch_size=5,  
                                                 color_mode='grayscale',
                                                 class_mode='categorical')
test_datagen=ImageDataGenerator(rescale=1./250,
								shear_range=0.2,
        						zoom_range=0.2,
        						horizontal_flip=True)

testing_set=test_datagen.flow_from_directory(r'C:\PROJECTS\SIGN_LANGUAGE_2\data\test',
												target_size=(64,64),
                                                 batch_size=5,  
                                                 color_mode='grayscale',
                                                 class_mode='categorical')
model.fit(
        training_set,
		steps_per_epoch=10,
		validation_data=testing_set,
        epochs=150,
		validation_steps=30,
		callbacks=[tensor_board]) 

model.save('CNN_SIGN_LANGUAGE')
