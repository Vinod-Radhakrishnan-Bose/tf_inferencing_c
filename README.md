# Tensorflow inferencing using C-APIs

This is a simple application for performing inferencing using TensorFlow C-APIs.

To run this application you will need to install the prebuilt Tensorflow libraries by following the steps described here:

https://www.tensorflow.org/install/lang_c

I installed the TensorFlow shared object (.so) files in /usr/local/lib and place the C-API file in my project folder. You can build using

```
gcc main.c inference_engine.c -I. -ltensorflow -o main
```

main.c is the main application which is using the inferencing functionality.

inference_engine.c/h contain all the utility functions for performing TensorFlow inferencing.

### Caveats: 

This is WIP. The current code is very specific to the model being used, and will need to be generalized.
