# DeepWarp
Tensorflow implement for DeepWarp: Photorealistic Image Resynthesis for Gaze Manipulation


## Instructions
This is a re-implement of paper [DeepWarp](https://sites.skoltech.ru/compvision/projects/deepwarp/) by Tensorflow. The results of my implement is slightly worse than the original paper which you can find it in [DeepWarp Demo Page](http://163.172.78.19/). Actually in general, I achieve the basic function for moving gaze. Some of the results and its drawbacks will be shown in behind.

But in the another hand, there are some differences between my implement and the paper.
- First, I use a more strict penalty for both the coarse warp part and the fine warp part which represents that I add the L2 distance between the output image of coarse warp and the ground truth image. In my experiment, I find that alone the regular L2 distance of synthesized image and final output image don't work. The L2 distance between the output image of coarse warp and the ground truth image is very large, which finally leads to a poor results to the output image. Based on this idea, I add a additional regular, and the weights is set to 1(I have tried some like 1e-0~1e-4, the first is the best). All of these details can be found in model.py, that is
```python
self.loss = self.coarse_loss + self.output_loss
```
- Second, I don't used the lightness correction module, because it can not bring any improvement. But I remains the module for future(because I find some similar results in eye white, see the fig3 of original paper).
- Third, I remove the input of local coordinate of landmarks. Due to some unknown reasons, it brings a better results. Actually, it brings more convenient that it reduced the influences of the algorithms of face detections(the step of warp the eyes).

For a better experience, I used some mess codes implementing a GUI to visual the results. All of these can be found in gui.py


## Update
2018.06.26, the first release version of DeepWarp, it finish the basic function for moving gaze.


## Experiment
The first one is the original image. The other two are the generated images. In the GUI, set the output path and click buttons vertical and horizontal to generate the two.
As you and see, the eye white is not very good. And also, when the value reach the boundary(vertical is [-30,30], horizontal is [-60, 60]) the results become unreasonable.

<p align="center"> 
	<img src="doc\obama.png" width="245", height="246">
	<img src="doc\horizontal.gif" width="245", height="246">
	<img src="doc\vertical.gif" width="245", height="246">
</p>