# Why Does This Repo Even Exist?

This repo was literally all about taking off the training wheels and having fun with coding once again. So, what better way than to implement a CNN without Python? 

Because... `import torch` hides all the beautiful and violent math that goes on behind the scenes. Let me tell you, it was not fun trying to debug this, and I genuinely should have kept a counter for the number of times I typed `#` instead of `//`... 

## More on Why This Repository Exists

If you have spent any time in modern machine learning, you know the feeling of no longer writing code... you are just mainly configuring high-level packages. Yes, coding is a part of it, but the "model training" we do is just `model.fit()`, watch a progress bar move, and call it a day. The models become black boxes, the math literally doesn't even cross our minds, and the thrill of software engineering is literally non-existent. 

I built this project to get lost in some code once again.
 
This is a CNN built entirely from scratch in raw C++... it felt good to give (your imagination) to TF, Torch, and Keras for once, because today I really felt like I was loving coding again. 

Without realizing it had become so natural for me, when I wanted to try the MNIST dataset, my brain was literally going blank. I was telling myself, just `tf.datasets.mnist` and call it a day. Just import `train_test_split`, get a good classifier from sklearn, and if you want to put in the effort, then use TF... but my goodness was it a RIDE to get this working on C++.

Getting lost between pointers and linear algebra... it once again felt like pure magic to me today, stripping away all the packages to redo the math. 

## Technical Stuff

Basically coded the entire forward pass and backward pass using C++ and Eigen. 

`Conv2D` -> `ReLU` -> `Max Pool` -> `Flatten` -> `Dense` -> `Sigmoid` and `Cross-Entropy`

## Insane Stuff That Happened

* Brain freeze when you realize you can't just `mnist.load_data()` and have to actually handle I/O.
* Max pool is not just a function to call. To remember winning pixels... took me a few days of YouTube videos to understand that once again. Going from the definition of "it just pools large values" to this... it's insane.
 
## Finally, the Results

**93.8% accuracy** (ADAM OPTIMIZER NEXT????)

## If You Want to Learn and Run This Code

1. Clone the repo.
2. Download MNIST (Kaggle dataset. Just check my path names; I have used the full path, but the final file name should match).
3. Ensure Eigen is present. I have linked it in `src/mnist/cnn_loader.hpp` at the top.
4. Compile with `g++`: 
```bash
   g++ src/main.cpp -I ./include/Eigen -O3 -o cnn_engine
```

##
Clarification
NO, this is not the most efficient way to run a CNN for production. It probably performs worse on time compared to Python libraries, but is it the most satisfying way... defo!!