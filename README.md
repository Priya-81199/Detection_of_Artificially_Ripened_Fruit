
# 
# Detection of Artificially Ripened Fruit using Transfer Learning

**TensorFlow** is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for *machine learning applications such as neural networks*.

**Flutter** is Google's SDK for crafting beautiful, fast user experiences for mobile, web and desktop from a **single codebase**. Flutter works with existing code, is used by developers and organizations around the world, and is *free* and *open source*.


## Dataset
The dataset is organised in **2 folders**. (Artifical and Natural)
This dataset contains **11,000 images of Bananas**.  


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
<a href="https://ibb.co/VV6kjG7"><img src="https://i.ibb.co/6YQqnS3/Whats-App-Image-2021-03-18-at-4-45-20-PM-1.jpg" alt="Whats-App-Image-2021-03-18-at-4-45-20-PM-1" border="0"></a>
<a href="https://ibb.co/tZFkRnc"><img src="https://i.ibb.co/yh7G2K5/Whats-App-Image-2021-03-18-at-4-45-20-PM.jpg" alt="Whats-App-Image-2021-03-18-at-4-45-20-PM" border="0"></a>
<a href="https://ibb.co/RBW9s3x"><img src="https://i.ibb.co/cCd1Z6G/Whats-App-Image-2021-03-18-at-5-04-54-PM.jpg" alt="Whats-App-Image-2021-03-18-at-5-04-54-PM" border="0"></a>


## Conclusion

Thus we have successfully predicted whether the banana is artificially ripened or not.

