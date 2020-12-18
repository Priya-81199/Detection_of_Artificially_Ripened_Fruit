import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:device_apps/device_apps.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

var time_in_ms = 0;

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
  String _model = 'thermal';
  bool val = false;

  @override
  void initState() {
    super.initState();
   onSelect(_model);
  }
  onSelect(model) {
      setState(() {
      _loading = true;
      _model = model;
      loadModel().then((value) {
        setState(() {
          _loading = false;
        });
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
              'Flower Vision',
              style:TextStyle(
                  fontFamily: 'Mono',
            )
          ),
        ),
        backgroundColor: Colors.deepPurple,

      ),
      body: _loading
          ? Center(
            child: Container(
                alignment: Alignment.center,
                child: CircularProgressIndicator(),
              ),
          )
          : Center(
            child: SingleChildScrollView(
              child: val?
              AlertDialog(
                title: Text("Which type of image to pick?"),
                actions: [
                  FlatButton(onPressed: () => { pickImage('thermal') }, child: Text("Thermal")),
                  FlatButton(onPressed: () => { pickImage('normal') }, child: Text("Normal"))
                ],

              )

              :Container(
        width: MediaQuery.of(context).size.width,
        child: Stack(
              //crossAxisAlignment: CrossAxisAlignment.center,
              //mainAxisAlignment: MainAxisAlignment.center,

            children: [
                _image == null ? Container() : AspectRatio(
                    aspectRatio: 0.77,
                    child: Image.file(_image)
                ),
                SizedBox(
                  height: 20,
                ),
                _outputs != null
                    ? Positioned(
                  left:10,
                  bottom: 100,
                      child: Text(
                      "${_outputs[0]["label"]} : "
                      "${_outputs[0]["confidence"]}\n"
                      "$time_in_ms ms",
                  style: TextStyle(
                    color: Colors.white,
                      fontSize: 25.0,
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
          ),
        floatingActionButton: Stack(
          children: <Widget>[
            Padding(
              padding: EdgeInsets.only(left:31),
              child: Align(
                alignment: Alignment.bottomLeft,
                child: FloatingActionButton(
                  onPressed: () => clickImage('normal'),
                  child: Icon(Icons.camera_alt),
                  backgroundColor: Colors.deepPurple,
                ),
              ),),

            Align(
              alignment: Alignment.bottomRight,
              child: FloatingActionButton(
                onPressed: () => showAlert1(),
                child: Icon(Icons.add_photo_alternate),
                backgroundColor: Colors.deepPurple,
              ),
            ),
            Padding(
              padding: EdgeInsets.only(left:31),
              child: Align(
                alignment: Alignment.bottomCenter,
                child: FloatingActionButton(
                  onPressed : () =>
                    thermalImage('thermal'),

                  child: Icon(Icons.camera),
                  backgroundColor: Colors.deepPurple,
                ),
              ),
            ),
          ],
        )
    );
  }


  void showAlert1() {
    setState(() {
    });
    val = true;
  }
  void reset(){
    setState(() {
    });
    val = false;
  }

  pickImage(String model) async {
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) return null;
    setState(() {
      _loading = true;
      _image = image;
    });
    await classifyImage(image,model);

  }
  clickImage(String model) async {
    var image1 = await ImagePicker.pickImage(source: ImageSource.camera);
    if (image1 == null) return null;
    setState(() {
      _loading = true;
      _image = image1;
    });
    await classifyImage(image1,model);
  }
  thermalImage(String model) async {
    DeviceApps.openApp('com.tyriansystems.SeekThermal');
    var image2 = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image2 == null) return null;
    setState(() {
      _loading = true;
      _image = image2;
    });
    await classifyImage(image2,model);
  }



  classifyImage(File image,String model) async {
    await reset();
    await onSelect(model);
    print("Classification started...");
    final stopwatch =  Stopwatch()..start();
    var output = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 2,
      threshold: 0.5,
      imageMean: 127.5,
      imageStd: 127.5,

    );

    time_in_ms = stopwatch.elapsedMilliseconds;
    print('classified in $time_in_ms');
    setState(() {
      _loading = false;
      _outputs = output;
    });


  }


  loadModel() async {
    dispose();
      String res;
      if (_model == 'normal') {
        print("Inside the normal model");
        res = await Tflite.loadModel(
          model: "assets/model_unquant.tflite",
          labels: "assets/labels.txt",
        );
      } else {
        print("Inside the thermal model");
        res = await Tflite.loadModel(
          model: "assets/model_unquant.tflite",
          labels: "assets/labels.txt",
        );
      }
      print(res);

  }
  @override
  void dispose() async {
    await Tflite.close();
    super.dispose();
  }
}




