
# 
# Detection of Artificially Ripened Fruit using Transfer Learning

**TensorFlow** is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for *machine learning applications such as neural networks*.

**Flutter** is Google's SDK for crafting beautiful, fast user experiences for mobile, web and desktop from a **single codebase**. Flutter works with existing code, is used by developers and organizations around the world, and is *free* and *open source*.

In this project we will develop a Flutter app that classifies 5 types of flowers using Transfer Learning in TensorFlow and shows the result to the user.![enter image description here](https://miro.medium.com/max/1020/1*qp84aqUzlD5dEVK9nTrj4Q.png)

**Currently due to unavailability of fruit ripening dataset we will use sample flower dataset.*


## Dataset
The dataset is organised in **5 folders**. Each folder contains flowers of one kind. The folders are named **sunflowers, daisy, dandelion, tulips and roses**. The data is hosted in a public bucket on *Google Cloud Storage.*

This dataset contains **3670 images of flowers**.  
![enter image description here](https://lh4.googleusercontent.com/8_jOXRV3Y2vBT72C1IdUz04pfSqWyP5Dkq5NvIFwn9fNhfC2XknnHKhCHvd-hISiEVNrLmNflYwPOHgpYcE1rwCA7Xqpw4HnbMfLimFOs-zl7nuA2OxdxhMi5tOlnDG_WCy4_Cer)![enter image description here](https://lh3.googleusercontent.com/og36ajmNenXx1SRP95HIGXZ7GlF06QoySLNpHFbgQD_bqUi7hBss84l4-M-90y0L49mvVosiPbFY5izlPHFDNh0o-t-jC2wun_DQvel1e4nGx8wWBPlJ7xCHlAQZpWzT7wEI_Yt4)![enter image description here](https://lh5.googleusercontent.com/jXellwfrqrvoGPJeMQM29EyaZfaAC3dO2rF1cmTlZxBG2IyfVNHFEZsDVEciGXVgBFaQJZkJ4Ss5kRAG8qwK6zUuGrI_Pgus0SqVbvLYIGpiyIniQ6qz9q9nl1-szrEBzT4IjMaK)![enter image description here](https://lh5.googleusercontent.com/J-lCrbQC6EQDulmGbhiUO1cg4LcaYqV-BOsGfN_UvnaFLHyzGBHsio1mUqYx_wR_KTDaa6_crVfjdqtyeQoQjcwN4lFSAw0Yp7JqMiB-yFvSm7YHIEBIlKAAFkVe2GhGBMgvqSLW)![enter image description here](https://lh3.googleusercontent.com/rsCQsIY26YvQpm67UfUX_yDGfE2JrBTg86EFfLZBJKpocOh3zVMwvbIjc_Z82HByLmok_TgYZ6RyLXoGmTAnKyKpGDsTY_7vHHP4Zivhn-YwucLD4H6aSw94qbdW6lAhWna1ClH1)![enter image description here](https://lh5.googleusercontent.com/z1PsDwOjFUhR16wAs_9CSrEsq_y479ZASzQUKti620KMraMGXRqUXtCl8sMGq6Hou5XuLjN1crBB747mMk_o-YW-u0K1a7UYoOPG_85EnN0p3guJsDTD3NqkGOS4iMCkudwhDTer)
```
gs://flowers-public/sunflowers/5139971615_434ff8ed8b_n.jpg
gs://flowers-public/daisy/8094774544_35465c1c64.jpg
gs://flowers-public/sunflowers/9309473873_9d62b9082e.jpg
gs://flowers-public/dandelion/19551343954_83bb52f310_m.jpg
gs://flowers-public/dandelion/14199664556_188b37e51e.jpg
gs://flowers-public/tulips/4290566894_c7f061583d_m.jpg
gs://flowers-public/roses/3065719996_c16ecd5551.jpg
gs://flowers-public/dandelion/8168031302_6e36f39d87.jpg
gs://flowers-public/sunflowers/9564240106_0577e919da_n.jpg
gs://flowers-public/daisy/14167543177_cd36b54ac6_n.jpg
```
### Loading Dataset faster ( Rather than importing image one by one)
### **The TFRecord file format**

Tensorflow's preferred file format for storing data is the  [protobuf](https://developers.google.com/protocol-buffers/)-based TFRecord format. Other serialization formats would work too but you can load a dataset from TFRecord files directly by writing:

```
filenames = tf.io.gfile.glob(FILENAME_PATTERN)
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...) # do the TFRecord decoding here - see below
```

For optimal performance, it is recommended to use the following more complex code to read from multiple TFRecord files at once. This code will read from N files in parallel and disregard data order in favor of reading speed.

```
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

filenames = tf.io.gfile.glob(FILENAME_PATTERN)
dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
dataset = dataset.with_options(ignore_order)
dataset = dataset.map(...) # do the TFRecord decoding here - see below
```

### **TFRecord cheat sheet**

Three types of data can be stored in TFRecords:  **byte strings**  (list of bytes), 64 bit  **integers** and 32 bit  **floats**. They are always stored as lists, a single data element will be a list of size 1. You can use the following helper functions to store data into TFRecords.

**writing byte strings**

```
# warning, the input is a list of byte strings, which are themselves lists of bytes
def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))
```

**writing integers**

```
def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))
```

**writing floats**

```
def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
```

**writing a TFRecord**, using the helpers above

```
# input data in my_img_bytes, my_class, my_height, my_width, my_floats
with tf.python_io.TFRecordWriter(filename) as out_file:
  feature = {
    "image": _bytestring_feature([my_img_bytes]), # one image in the list
    "class": _int_feature([my_class]),            # one class in the list
    "size": _int_feature([my_height, my_width]),  # fixed length (2) list of ints
    "float_data": _float_feature(my_floats)       # variable length  list of floats
  }
  tf_record = tf.train.Example(features=tf.train.Features(feature=feature))
  out_file.write(tf_record.SerializeToString())
```

To read data from TFRecords, you must first declare the layout of the records you have stored. In the declaration, you can access any named field as a fixed length list or a variable length list:

**reading from TFRecords**

```
def read_tfrecord(data):
  features = {
    # tf.string = byte string (not text string)
    "image": tf.io.FixedLenFeature([], tf.string), # shape [] means scalar, here, a single byte string
    "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar, i.e. a single item
    "size": tf.io.FixedLenFeature([2], tf.int64),  # two integers
    "float_data": tf.io.VarLenFeature(tf.float32)  # a variable number of floats
  }

  # decode the TFRecord
  tf_record = tf.parse_single_example(data, features)

  # FixedLenFeature fields are now ready to use
  sz = tf_record['size']

  # Typical code for decoding compressed images
  image = tf.image.decode_jpeg(tf_record['image'], channels=3)

  # VarLenFeature fields require additional sparse.to_dense decoding
  float_data = tf.sparse.to_dense(tf_record['float_data'])

  return image, sz, float_data

# decoding a tf.data.TFRecordDataset
dataset = dataset.map(read_tfrecord)
# now a dataset of triplets (image, sz, float_data)
```

Useful code snippets:

**reading single data elements**

```
tf.io.FixedLenFeature([], tf.string)   # for one byte string
tf.io.FixedLenFeature([], tf.int64)    # for one int
tf.io.FixedLenFeature([], tf.float32)  # for one float
```

**reading fixed size lists of elements**

```
tf.io.FixedLenFeature([N], tf.string)   # list of N byte strings
tf.io.FixedLenFeature([N], tf.int64)    # list of N ints
tf.io.FixedLenFeature([N], tf.float32)  # list of N floats
```

**reading a variable number of data items**

```
tf.io.VarLenFeature(tf.string)   # list of byte strings
tf.io.VarLenFeature(tf.int64)    # list of ints
tf.io.VarLenFeature(tf.float32)  # list of floats
```

A VarLenFeature returns a sparse vector and an additional step is required after decoding the TFRecord:

```
dense_data = tf.sparse.to_dense(tf_record['my_var_len_feature'])
```

It is also possible to have optional fields in TFRecords. If you specify a default value when reading a field, then the default value is returned instead of an error if the field is missing.

```
tf.io.FixedLenFeature([], tf.int64, default_value=0) # this field is optional
```

## Training the model
### Transfer Learning

For an image classification problem, dense layers will probably not be enough. We have to learn about convolutional layers and the many ways you can arrange them.

But we can also take a shortcut! There are fully-trained convolutional neural networks available for download. It is possible to chop off their last layer, the softmax classification head, and replace it with your own. All the trained weights and biases stay as they are, you only retrain the softmax layer you add. This technique is called transfer learning and amazingly, it works as long as the dataset on which the neural net is pre-trained is "close enough" to yours.

With transfer learning, you benefit from both advanced convolutional neural network architectures developed by top researchers and from pre-training on a huge dataset of images. In our case we will be transfer learning from a network trained on ImageNet, a database of images containing many plants and outdoors scenes, which is close enough to flowers.
    ![enter image description here](https://lh3.googleusercontent.com/eH3zQS3oQuyEZciGgss6Iv7B7pUnF_Oq3vpWqo4-JCT7ZunE1raDlnbkkozjadL19625wqLGnIXv9rxaAQtqJ3tFVriWJspe_PLtxCorqGdW_xeepcetSppbwonDr1TUF7mlvdEQ)

### **Transfer learning in Keras**

In Keras, you can instantiate a pre-trained model from the  `tf.keras.applications.*`  collection. MobileNet V2 for example is a very good convolutional architecture that stays reasonable in size. By selecting  `include_top=False`, you get the pre-trained model without its final softmax layer so that you can add your own:

```
pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
pretrained_model.trainable = False

model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='softmax')
])
```
![enter image description here](https://lh6.googleusercontent.com/R9PjsenJH05ZPQ5avjtotPRlEygOMzI-DYQ_w_0iJxZINMOPgDSqZyhTPjI4CoccAUVOTy48xWL-KqRKgy70RA1CQszQCji340OANsQXKmzKKEV2k3Aqzideb45L11MZ4GQnwIBo)
Also notice the  `pretrained_model.trainable = False`  setting. It freezes the weights and biases of the pre-trained model so that you train your softmax layer only. This typically involves relatively few weights and can be done quickly and without necessitating a very large dataset. However if you do have lots of data, transfer learning can work even better with  `pretrained_model.trainable = True`. The pre-trained weights then provide excellent initial values and can still be adjusted by the training to better fit your problem.

Finally, notice the  `Flatten()`  layer inserted before your dense softmax layer. Dense layers work on flat vectors of data but we do not know if that is what the pretrained model returns. That's why we need to flatten. In the next chapter, as we dive into convolutional architectures, we will explain the data format returned by convolutional layers.

## Convert the Model to TensorFlow Lite File with Optimization

TensorFlow Lite provides tools to optimize the size and performance of your models, often with minimal impact on accuracy. Optimized models may require slightly more complex training, conversion, or integration.

Machine learning optimization is an evolving field, and TensorFlow Lite's  [Model Optimization Toolkit](https://www.tensorflow.org/lite/guide/get_started#model_optimization_toolkit)  is continually growing as new techniques are developed.

### Performance

The goal of model optimization is to reach the ideal balance of performance, model size, and accuracy on a given device.  [Performance best practices](https://www.tensorflow.org/lite/performance/best_practices)  can help guide you through this process.

### Quantization

By reducing the precision of values and operations within a model, quantization can reduce both the size of model and the time required for inference. For many models, there is only a minimal loss of accuracy.

The TensorFlow Lite converter makes it easy to quantize TensorFlow models. 
```
model.save('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
```

    tflite_model_file = pathlib.Path('model.tflite')
    tflite_model_file.write_bytes(tflite_model)

TensorFlow Lite supports reducing precision of values from full floating point to half-precision floats (float16) or 8-bit integers. There are trade-offs in model size and accuracy for each choice, and some operations have optimized implementations for these reduced precision types.






## Add TensorFlow Lite to the Flutter app

Copy the TensorFlow Lite model **model.tflite** and **label.txt** that you trained earlier to assets folder at **flutter_project/assets/**.

![enter image description here](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/img/4081e81d912ce7c4.png)



## Installing tflite, Image_picker package and tflite_flutter packages.

### 1.  **tflite package**

Add `tflite` as a dependency in your pubspec.yaml file.
	 
	 dependencies: 
			 tflite: any
You can install packages from the command line with Flutter:

	$  flutter pub get

Now in your Dart code, you can use:

    import  'package:tflite/tflite.dart';

#### Android
In  `android/app/build.gradle`, add the following setting in  `android`  block.
               
    aaptOptions {  
      noCompress "tflite"  
                }
### 2.  **Image_picker Package**

First, add `image_picker` as a dependency in your pubspec.yaml file

Add this to your package's pubspec.yaml file:

    dependencies:  
	    image_picker: ^0.6.6+1
You can install packages from the command line:

    $  flutter pub get
Now in your Dart code, you can use:

    import  'package:image_picker/image_picker.dart';
### 3. tflite_flutter

Add this to your package's pubspec.yaml file:

```yaml
dependencies:
  tflite_flutter: ^0.5.0
```

You can install packages from the command line:
with Flutter:

```shell
$ flutter pub get
```

Alternatively, your editor might support  `flutter pub get`. Check the docs for your editor to learn more.

Now in your Dart code, you can use:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
```
## Accelerate inference with GPU delegate
TensorFlow Lite supports several hardware accelerators to speed up inference on your mobile device. [GPU](https://www.tensorflow.org/lite/performance/gpu) is one of the accelerators that TensorFlow Lite can leverage through a delegate mechanism and it is fairly easy to use.

**Android**  GpuDelegateV2

```dart
final gpuDelegateV2 = GpuDelegateV2(
        options: GpuDelegateOptionsV2(
        false,
        TfLiteGpuInferenceUsage.fastSingleAnswer,
        TfLiteGpuInferencePriority.minLatency,
        TfLiteGpuInferencePriority.auto,
        TfLiteGpuInferencePriority.auto,
    ));

var interpreterOptions = InterpreterOptions()..addDelegate(gpuDelegateV2);
final interpreter = await Interpreter.fromAsset('your_model.tflite',
    options: interpreterOptions);
```
## Code in main.dart

Designed Interactive Flutter app in  **main.dart**.

## Running on Emulator/Android Device

After the code completion, run the app on emulator/Andriod Device
1. For starting emulator click on the AVD manager on the top right corner of Android Studio. Further select the emulator listed there.
2. For Andriod device make sure that USB debugging is ON in Developers options(Phone>Settings>Developer Options>USB Debugging).


## Result

Here are some screenshots of the working app.


## Conclusion

Thus we have successfully executed the Flower Vision app using Flutter.

