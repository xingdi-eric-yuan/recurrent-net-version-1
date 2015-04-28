#recurrent-net-version-1.0
=====================

Simplest uni-directional recurrent neural network (C++ / OpenCV).

To run this code, you should have 
* OpenCV.

##Compile & Run
* Compile by running:
```
cmake .
make
```
* Run: 
```
./rnn
```
##Structure and Algorithm
See [my tech-blog](http://eric-yuan.me/rnn1/).

##TODO
...

##Config Files

####General Parameters Config
* gradient checking (automatically disable dropout, and use tiny dataset)
* store parameters into log file (for debugging, should be faster if not using it)
* batch size
* non-linearity method (sigmoid, tanh, ReLU)
* training epochs
* iteration per epoch
* learning rate
* ngram
* training percent (for cross validation)

####Layers Config
* layer type
* amount of hidden neurons
* dropout rate
* weight decay
* amount of output classes (in **Softmax** Layer, initialized to be 0, because it depends on dataset)

####Multi-Layer
This network supports multiple hidden layers.

The MIT License (MIT)
------------------

Copyright (c) 2015 Xingdi (Eric) Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.