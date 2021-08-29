# Dminity

Train a YOLOX model to detect home amenities from images.

Significance: Small model size, Fast inference time, High accuracy

## Dataset 

OpenImageV6 with 30 labels for **object detection** task:

```
Toilet, Swimming_pool, Bed, Billiard_table, Sink, Fountain, Oven, Ceiling_fan,
Television, Microwave_oven, Gas_stove, Refrigerator,
Kitchen_&_dining_room_table, Washing_machine, Bathtub, Stairs, Fireplace,
Pillow, Mirror, Shower, Couch, Countertop, Coffeemaker, Dishwasher, Sofa_bed,
Tree_house, Towel, Porch, Wine_rack, Jacuzzi 
```

## Timeline

- [ ] Start small: We start with the end-to-end process with only 1 label first
  (Bathtub).
  - [X] Download OpenImageV6 for class Bathtub (use this
    [notebook](https://colab.research.google.com/drive/14ISeuv3frabPFo2F-giIzZdPr2dukmLW#scrollTo=tzyrJovZPa3I))
  - [X] Upload to Roboflow (TODO: YOLOX requires what image size?)
  - [X] Export as link
  - [ ] Use the Roboflow YOLOX training notebook to train the model
  - [ ] Download the model and do inferences
  - [ ] Setup either Weight & Biases or Tensorboard (edit the original YOLOX
    training notebook)
  - [ ] Setup experiment environment with 3 variants of the YOLOX model (yolo-s,
    yolo-m, yolox)

- [ ] Experiment with 10 classes (to choose a model variant)
  - [X] Download OpenImageV6 for the 10 classes (use this
    [notebook](https://colab.research.google.com/drive/14ISeuv3frabPFo2F-giIzZdPr2dukmLW#scrollTo=tzyrJovZPa3I))
  - [ ] Upload to Roboflow
  - [ ] Export as link
  - [ ] Record the experiment result and make conclusion based on each variant's
    pros and cons

- [ ] Experiment with 200 images of all 30 classes
  - [X] Download OpenImageV6 for the all 30 classes and limit to 200 (use this
    [notebook](https://colab.research.google.com/drive/14ISeuv3frabPFo2F-giIzZdPr2dukmLW#scrollTo=tzyrJovZPa3I))
  - [ ] Upload to Roboflow
  - [ ] Export as link
  - [ ] Record the experiment result and make conclusion based on each variant's
    pros and cons

- [ ] Train with all 30 labels with the chosen model from last step
  - [ ] Download images for all 30 classes
  - [ ] Upload to Roboflow
  - [ ] Export as link
  - [ ] Train
  - [ ] Validate
  - [ ] Download model

- [ ] Create an MVP application (TODO)


## Log

### 2020-08-29

Created a
[notebook](https://colab.research.google.com/drive/14ISeuv3frabPFo2F-giIzZdPr2dukmLW#scrollTo=tIE5_pB4IeG6)
Download Custom OpenImage Dataset and Upload to Google Drive.

Uploaded custom dataset with 1 class -- Bathtub -- to roboflow.

The
[notebook](https://colab.research.google.com/drive/1eZk39KM8PubtwisTqWk_L-RT6c_ARN_K#scrollTo=s5h536amH32Z)
for training YOLOX with roboflow requires Pascal VOC export format.

[Here](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/manipulate_training_image_size.md)
it says that YOLOX needs 640x640 input size. Yolo-tiny and Yolo-nano needs
416x416.

Tomorrow: 
- Download the 10 classes dataset and upload to drive
- Resize the Bathtub dataset to what YOLOX requires and continue with training
  and setting up for experimentation.
