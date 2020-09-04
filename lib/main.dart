import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:device_apps/device_apps.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(MaterialApp(
  home: MyApp(),
  debugShowCheckedModeBanner: false,
));

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  List _outputs;
  File _image;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _loading = true;

    loadModel().then((value) {
      setState(() {
        _loading = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        resizeToAvoidBottomPadding: false,
      appBar: AppBar(
        title: Center(
          child: const Text(
              'FLOWER VISION',
              style:TextStyle(
                  fontFamily: 'Mono',
            )
          ),
        ),
        backgroundColor: Colors.teal,

      ),
      body: _loading
          ? Container(
              alignment: Alignment.center,
              child: CircularProgressIndicator(),
            )
          : SingleChildScrollView(
            child: Container(
        width: MediaQuery.of(context).size.width,
        child: Stack(
            //crossAxisAlignment: CrossAxisAlignment.center,
            //mainAxisAlignment: MainAxisAlignment.center,

          children: [
              _image == null ? Container() : AspectRatio(
                  aspectRatio: 0.69,
                  child: Image.file(_image)),
              SizedBox(
                height: 20,
              ),
              _outputs != null
                  ? Positioned(
                left:60,
                bottom: 180,
                    child: Text(
                "${_outputs[0]["label"]}",
                style: TextStyle(
                  color: Colors.white60,
                    fontSize: 80.0,
                    fontFamily: 'Mono',
                    background: Paint()..color = Colors.black45,
                ),
              ),
                  )
                  : Container(),
            ],
        ),
      ),
          ),
        floatingActionButton: Stack(
          children: <Widget>[
            Padding(
              padding: EdgeInsets.only(left:31),
              child: Align(
                alignment: Alignment.bottomLeft,
                child: FloatingActionButton(
                  onPressed: clickImage,
                  child: Icon(Icons.camera_alt),
                  backgroundColor: Colors.teal,
                ),
              ),),

            Align(
              alignment: Alignment.bottomRight,
              child: FloatingActionButton(
                onPressed: pickImage,
                child: Icon(Icons.add_photo_alternate),
                backgroundColor: Colors.teal,
              ),
            ),
            Padding(
              padding: EdgeInsets.only(left:31),
              child: Align(
                alignment: Alignment.bottomCenter,
                child: FloatingActionButton(
                  onPressed: thermalImage,
                  child: Icon(Icons.camera),
                  backgroundColor: Colors.teal,
                ),
              ),
            ),
          ],
        )
    );
  }

  pickImage() async {
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) return null;
    setState(() {
      _loading = true;
      _image = image;
    });
    classifyImage(image);
  }
  clickImage() async {
    var image1 = await ImagePicker.pickImage(source: ImageSource.camera);
    if (image1 == null) return null;
    setState(() {
      _loading = true;
      _image = image1;
    });
    classifyImage(image1);
  }
  thermalImage() async {
    DeviceApps.openApp('com.tyriansystems.SeekThermal');
    var image2 = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image2 == null) return null;
    setState(() {
      _loading = true;
      _image = image2;
    });
    classifyImage(image2);
  }



  classifyImage(File image) async {

    var output = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 2,
      threshold: 0.5,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    setState(() {
      _loading = false;
      _outputs = output;
    });
  }

  loadModel() async {

    await Tflite.loadModel(
      model: "assets/model_unquant.tflite",
      labels: "assets/labels.txt",
    );
    set_backend();
  }
  set_backend() async{

    final gpuDelegateV2 = GpuDelegateV2(
        options: GpuDelegateOptionsV2(
          false,
          TfLiteGpuInferenceUsage.fastSingleAnswer,
          TfLiteGpuInferencePriority.minLatency,
          TfLiteGpuInferencePriority.auto,
          TfLiteGpuInferencePriority.auto,
        ));

    var interpreterOptions = InterpreterOptions()..addDelegate(gpuDelegateV2);
    final interpreter = await Interpreter.fromAsset('model_unquant.tflite', options: interpreterOptions);
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }
}


