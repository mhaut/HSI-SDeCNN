# A Single Model CNN for Hyperspectral Image Denoising
The Code for "A Single Model CNN for Hyperspectral Image Denoising". []
```
Alessandro Maffei, Juan M. Haut, Mercedes E. Paoletti, Javier Plaza, Lorenzo Bruzone and Antonio Plaza.
A Single Model CNN for Hyperspectral Image Denoising.
IEEE Transactions on Geoscience and Remote Sensing.
DOI: 
Submitted, 2019.
```

![HSI-SDeCNN](https://github.com/mhaut/HSI-SDeCNN/blob/master/images/denspectral.png)



## Example of use

### Some tips

```
In the "trainset" folder are reported the 4 part of the Washigton DC mall HSI used for the training of the network, while in the "testsets" directory the HSI on which the method has been tested.

To train the network run "Demo_training_HSI_SDeCNN.mat". The best model will be saved in the "data" folder.

To apply the denoising process with the model trained in the paper (the one in the "BestModel" directory) run the demo-test scripts. The results will be saved in the "Results" folder.

NB! if you want to change the number of considered channel in the training process you need to change the "nch" variable in the "model_train.mat" and in the "model_init_HSI_SDeCNN.mat" scripts.
```

### Reference code

```
https://github.com/cszn/FFDNet
```

