# D-NeuS
**Recovering Fine Details for Neural Implicit Surface Reconstruction (WACV2023)**

![scan37](gifs/scan37.gif)

![scan110](gifs/scan110.gif)

## Code
D-NeuS is built on [NeuS](https://github.com/Totoro97/NeuS). It improves the surface reconstruction quality with fine details recovered.

### Training
```shell
python exp_runner.py --mode train --case <e.g., scan24>
```
### Extract surface from trained model

```shell
python exp_runner.py --mode validate_mesh --case <e.g., scan24> --is_continue
```
### View interpolation

```shell
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --case <e.g., scan24> --is_continue 
```


## Data 
Download the pre-processed DTU data from this [link](https://drive.google.com/drive/folders/16aL9nbFss4Jw5tBw5B1FLnTjgRQJltCl?usp=sharing), and put it under ./data folder.


## Acknowledgement
The code is based on [NeuS](https://github.com/Totoro97/NeuS). Some code snippets and data pre-processing are borrowed from [MVSDF](https://github.com/jzhangbs/MVSDF). Thanks for these great works.

