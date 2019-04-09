Supplementary material for [this issue on pytorch](https://github.com/pytorch/pytorch/issues/7786).

---

This repository now contains an implementation in pytorch of [goodfellow's trick](https://arxiv.org/pdf/1510.01799.pdf) to get individual gradients from a minibatch for a feedforward network.

The current implementation uses relu activation functions and for a binary classification issue, but this can be easily changed in `models.py` and does not impact the backward pass made in `goodfellow_backprop.py`.

*Note*: While the title says `with-autodiff` (what I wanted), the code here does not do that.
Instead, it is rewriting the backward pass so that the individual gradients are not summed.
The code here only works for feedforward/MLP/linear layers with nonlinearities, and adapting it to different layers (CNN/RNN) would require quite a bit of work.

---

### Context:

Given a neural network and minibatch of samples, how can we compute gradients for _each sample_ quickly?

Pytorch's interface does not give a simple and efficient way to achieve this, but for "simple enough" machine learning models (including feedforward neural nets), it is possible to use [this trick](https://arxiv.org/pdf/1510.01799.pdf) by Ian Goodfellow (if implemented efficiently, which I didn't at the start and led me to open [this issue](). _Lesson #1: [batch your matrix multiplications](https://pytorch.org/docs/master/torch.html#torch.bmm)_).

---

### How/Why the trick described [here](https://arxiv.org/pdf/1510.01799.pdf) works

(I haven't found how to do math/latex in github yet, sorry)

---

#### Notation:

* `X_l` : Inputs at layer `l`

  input data for `X_0`, result of activation of previous layer otherwise.

* `W_l` : Weights at layer `l`

* `C_l` : Linear combinations at layer `l`,  `C_l = X_l W_l`


So that the network looks like

    X_0 -> C_0 -> X_1 -> C_1 -> X_2 -> ... -> X_L

where going from `X_l` to `C_l` involves multiplying by `W_l` and going from `C_l` to `X_{l+1}` involve passing `C_l` through the activation function.

---

#### Computing the gradients

When the backward pass eventually reaches layer `l`, which computes `C_l = X_l W_l`, the derivative of the function with respect to `C_l` (which I'll write `G_l`) is known.

Since the layer is just a linear function, the formula to compute the derivative with respect to `W_l` is simply `X_l^T G_l`, and this is where the summation/averaging over samples takes place.

If `X_l^T` is a `[D_l x N]` matrix and `G_l` is a `[N x D_{l+1}]` matrix, the result is `[D_l x D_{l+1}]` (the size of `C_l`), and has been summed up over the `N` dimension.

To get the individual gradient for the `n`th sample, we would need to compute the outer product `X_l[n,:] G_l[n,:]^\top`.

It is possible to implement this in an efficient way using Pytorch's [`bmm` function](https://pytorch.org/docs/master/torch.html#torch.bmm). 

Don't try to do a for loop over the `N` samples and computer the outer products using [`ger`](https://pytorch.org/docs/master/torch.html#torch.ger), it will be way too slow. 

If the inputs to Pytorch's `bmm` function are matrices of shapes `[N x D_in x 1]` and `[N x 1 x D_out]`, it will return a `[N x D_in x D_out]` matrix where each of the `N` dimension contains the gradient for one sample.

---

#### Goodfellow's trick for a feedforward network

The idea is:
* During the forward pass, we store the activations `X_l` and linear combinations `C_l` along with the final output of the model.
  
  (Done in the [`forward`](https://github.com/fKunstner/fast-individual-gradients-with-autodiff/blob/master/pytorch/models.py#L23) method of the model)
  
* For the backward pass, instead of asking Pytorch for the gradient with respect to `W_l`, we ask for the gradient w.r.t. `C_l`, such that it returns us the matrices `G_l`. 

  (Done in the [`grad_funcs.goodfellow`](https://github.com/fKunstner/fast-individual-gradients-with-autodiff/blob/master/pytorch/gradient_funcs.py#L39) function)

* We can now use `X_l` and `G_l` to compute the individual gradients using `bmm`

  (Done in the [`goodfellow_backprop`](https://github.com/fKunstner/fast-individual-gradients-with-autodiff/blob/master/pytorch/goodfellow_backprop.py#L4) function )

---

### Performance

Computing individual gradients for a minibatch of size ~100 for even "big" networks (50 layers of 300 units each) is only ~10x slower than computing the summed gradient, compared to the naive method of computing the gradient for each sample by repeatedly calling backwards, which can be ~50-100x slower.

The [`main.py`](https://github.com/fKunstner/fast-individual-gradients-with-autodiff/blob/master/pytorch/main.py) file runs the benchmarks if you want to try it on your machine for your network architecture, and the output for some setups on my machine is available [here](https://github.com/fKunstner/fast-individual-gradients-with-autodiff/blob/master/pytorch/results.txt).

---

### Side note:

It is also possible to redefine the backward pass of the linear function, which is the only one that has parameters in the feedforward neural network case, as shown how to do in [this post](https://github.com/pytorch/pytorch/issues/7786#issuecomment-391637797) by [Adam Paszke]https://github.com/apaszke). 

However, he is not advocating to do this, and I don't think you should - he was showing me how to do it as I found out you could do that and asked about it, but it is really hacky.

It seems cleaner to do it outside of the backward pass if you can, as it might break other stuff relying on the `backward` function.


---

### Thanks!

Code prepared by Frederik Kunstner, Aaron Mishkin and Didrik Nielsen.

Many thanks to the commenters on the [Pytorch issue](https://github.com/pytorch/pytorch/issues/7786#issuecomment-391612473),
especially [Adam Paszke](https://github.com/apaszke) and [Thomas Viehmann](https://github.com/t-vi) who have been very helpful in understanding what's going on.
