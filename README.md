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
  - [X] Use the Roboflow YOLOX training notebook to train the model
  - [ ] Download the model and do inferences
  - [X] Setup either Weight & Biases or Tensorboard (edit the original YOLOX
    training notebook)
  - [X] Setup experiment environment with 3 variants of the YOLOX model (yolo-s,
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

<details>

  <summary>2020-08-29 - Environment setup</summary>

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
  - [X] Download the 10 classes dataset and upload to drive
  - [X] Resize the Bathtub dataset to what YOLOX requires and continue with training
  - [X] setting up for experimentation.

</details>


<details>

  <summary>2020-08-31 - Trained with 1 class and setup for experimentation</summary>

  Trained using the notebook with the bathtub dataset. Confirmed that 640x640 is
  the correct input size for the model.

  The eval cell does'nt work, with an error of division with zero (the zero is the
  number of eval images, the `n_samples`). However, the folder containing eval
  list of images in `/content/YOLOX/datasets/VOC2012/ImageSets/Main/val.txt` does
  have a lot of items. TODO: look into the evaluator dataset loader script.

  Things to note: the train & test were successful although they're using the same
  dataloader.

  ```
  File "/content/YOLOX/yolox/evaluators/voc_evaluator.py", line 167, in evaluate_prediction
      a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
                            │                 │           │    │          └ 64
                            │                 │           │    └ <torch.utils.data.dataloader.DataLoader object at 0x7fc4a27cdad0>
                            │                 │           └ <yolox.evaluators.voc_evaluator.VOCEvaluator object at 0x7fc4a27cd8d0>
                            │                 └ 0.0
                            └ 0.0

  ZeroDivisionError: float division by zero
  ```

  The first inference result (yolox-s):

  ![](./imgs/first_inference_eval.png)

#### Experimentation

  For the experimentation, I saw somewhere in the trainer that it writes to a
  tensorboard's `SummaryWriter`. If I can load it into tensorboard locally, I can
  see the result after training finished for the day & make conclusions.

  Path of the trainer: `/content/YOLOX/yolox/core/trainer.py`

  On line 178:

  ```Python
    # Tensorboard logger
    if self.rank == 0:
        self.tblogger = SummaryWriter(self.file_name)
  ```

  The Tensorboard events are stored in the experiment folder.

  Path: `/content/YOLOX_Outputs/<experiment_name>`

  Just zip the whole directory, mount gdrive and copy. Then, download to local and
  launch tensorboard locally to see the experiment.

  The trainer only writes the average precision though, idk if there are other
  useful information to get.

  TODO: check other information one can get from `tensorboard.SummaryWriter`

  ![](./imgs/tensorboard_bathtub_yolox-s.png)

  To experiment with other yolox variants, 

  1. Download pretrained weights from [the checkpoint storage](https://github.com/Megvii-BaseDetection/storage/releases) 
  2. Copy dataloaders from the yolo_s example into `exps/default/<model_variant>`
     to make the train script load Pascal VOC format datasets
  3. Train
  4. Zip and download outputs
  5. Watch output in tensorboard locally

  **Tensorboard with the training outputs of yolox-s and yolox-m for bathtub:**

  ![](./imgs/tensorboard_bathtub_yolox-s-and-m.png)

</details>


<details>

  <summary>2020-09-02 - Download 200 training images from each class</summary>

  08:30 -- Tried uploading the first 10 classes to roboflow with no limit (16275 train,
  1124 test, 280 validation). Browser kept going out of memory.

  However, roboflow has "add more image to a dataset" feature, so maybe we can
  upload 5 classes at a time for 30/5 = 6 times. Could take a day or two of
  downloading and uploading.

  That's a problem for when we want to do the end-to-end process with all the
  data. But today, we are going to do 10 classes at a time with a limit of 200
  data per class to start with experimentation.

  10:02 -- Uploading `OID_lim200_1-10` to roboflow (classes 1 through 10,
  limited to 200 images for each class.) and downloading `OID_lim200_11-20` from
  OpenImage.

  10:31 -- `OID_lim200_01-10` had some problem (didnt map the class code to
  class name). Need to redownload. `OID_lim200_11-20` works fine with a total of
  2479 images.

  12:41 -- Downloaded `OID_lim200_01-10`, `OID_lim200_11-20` &
  `OID_lim200_21-30` and created a new project on roboflow just for
  experimenting

  Here are all the number of images in each 10-class-bucket limit to 200 images
  each class that we need to upload:

  ```shell
  ╰─ ls OID_lim200_01-10/Dataset/*/*/*.jpg | wc -l
  2825

  ╰─ ls OID_lim200_11-20/Dataset/*/*/*.jpg | wc -l
  2479

  ╰─ ls OID_lim200_21-30/Dataset/*/*/*.jpg | wc -l
  1719
  ```

  And the size of each bucket:

  ```shell
  ╰─ du -hs OID_lim200_*
  1019M   OID_lim200_01-10
  886M    OID_lim200_11-20
  718M    OID_lim200_21-30
  ```

  14:17 -- Done for the day.

</details>
