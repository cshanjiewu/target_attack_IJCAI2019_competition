# IJCAI Competition 2019
There is my submission at IJCAI 2019 competition. I ranked 6th on target attack track.

### Competition Description

- **Dataset**:About 110,000 product images of 110 categories from the e-commerce platform. Each image belongs to a specific category. 
- **Attack Goal in Targeted Attack**: It requests the attack model to generate an adversarial image, not only fool the defense model, but also misclassify it to a given specific label.
- **Submission**: Build a docker image and upload the docker image

### Targeted Adversarial Attack(6th Place)

 My solution is based on [NIPS 2017 1st solution in targeted attack track][1]. And I did some modifications. There are some useful modifications to increase scores.

1. Using the ensemble model to simulate the black box model. I used vgg16, inception_v1, resnetv1_50, resnetv2_152, inception_v3.
2. Using random pixel dropout to increase the generalization of adversarial examples. In the beginning of each iteration, I randomly generate a mask to blend some specific pixel to the original pixel value.
3. Using random input diversity. input will be randomly resizing and padding in each iterations.
4. Using Perlin noise to guide the direction of finding the adversarial example. Perlin noise is a low-frequency noise and the low frequency noise can be removed hardly.
5. Calculating each loss of simulated model separately.



[1]: https://github.com/dongyp13/Targeted-Adversarial-Attack	"nips2017"

