# Multi-View Azimuth Stereo via Tangent Space Consistency

[Xu Cao](https://hoshino042.github.io/homepage/), [Hiroaki Santo](https://sites.google.com/view/hiroaki-santo/), [Fumio Okura](http://cvl.ist.osaka-u.ac.jp/user/okura/) and [Yasuyuki Matsushita](http://www-infobiz.ist.osaka-u.ac.jp/en/member/matsushita/)

[CVPR 2023](https://cvpr2023.thecvf.com)

Paper and more details coming soon.

# Quick Start

```commandline
git clone https://github.com/xucao-42/mvas.git
cd mvas

conda env create -f environment.yml 
conda activate mvas

mkdir data
```
Download data from [Google Drive](https://drive.google.com/file/d/1C4Uf00nW-quKf_3YGD1M86AIsDMQ8Cj6/view?usp=sharing) and extract it under `data` folder.
Optimize the shape using azimuth maps:
```
cd code
python exp_runner.py --config configs/diligent_mv.conf # run
```

Results will be saved in `results/$obj_name/$exp_time`.

[//]: # (# Data folder structure)

[//]: # ()
[//]: # (```)

[//]: # (- input_azimuth_maps: 16-bit RGBA images. The alpha chanel is the object mask. The RGB channels are identical.)

[//]: # (Each channel can be multiplied by pi/65535 to obtaine the azimuth map in the range [0, pi]. )

[//]: # (Note we do not have to store the azimuth map within the range [0, 2pi] since our method is pi-invariant, i.e., our method treats a and a+pi samely.)

[//]: # (- vis_azimuth_maps: for pleasing visualization.)

[//]: # (- normal_maps : the normal map we used to compute the input azimuth map.)

[//]: # (-)

[//]: # (```)
