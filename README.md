# Deep autofocus (key frame detection)


Recording objects right on the focus is fundamental/important in microscopy. However, not only the stage will drift away when the measurement gets longer, but also the lateral movement will cause the drift on the axial direction. The help from machine to automatically correct the focal place is indispensable for the long-term measurement.


## Description of workflow

In this work, the correct focal plan can be determined by machine through the consecutive images in axial direction. To be more specific, the full operation is separated into two parts, coarse and fine adjustment. For the coarse adjustment, three snapshot images collected from corresponding three axial-positions, [-1, 0, 1] micrometer, which are related to the current one were the input data for machine to distinguish the focused one among three. Next, the seven snapshot images from seven finer axial-positions [-750, -500, -250, 0, 250, 500, 750] nanometer relative to the position chosen in coarse were for machine to select the most possible one in the fine adjustment.
The reasons behind such operation were three: 1. As humans, the "right" focal place was found through the comparison between other planes. Moreover, in most of cases, such plane was determined by coarse and fine adjustments of the stage. 2. The second operation, fine adjustment, could possibly correct the focal plane if the machine selected a wrong axial position. 3. In the time-lapse experiment, a smooth change in the focal plan is ideal for presentation.


## Collection of training data

The amount of data is key to make machine work. Instead of manually finding the cell and recording different axial positions, the use of cell mask in this project (https://github.com/tehsinchen/Eff-Unet-keras) to determine and centralize the cell significantly reduced the effort and time. Below is the example of raw images and corresponding label (key frame) as the training data. 


### axial_coarse:
![image](https://github.com/tehsinchen/deep_autofocus/blob/main/axial_coarse/data/axial_coarse.gif)
 label: [0, 1, 0]

### axial_fine:
![image](https://github.com/tehsinchen/deep_autofocus/blob/main/axial_fine/data/axial_fine.gif)
 label: [0, 0, 1, 1, 1, 0, 0]

Notice that the label contains three ones. This is because it is very hard to determine the best focal plane within +/- 250 nm (even for humans). Therefore, the roughness was included in the model.


## Result
![image](https://github.com/tehsinchen/deep_autofocus/blob/main/result_demo/demo_axial.gif)
