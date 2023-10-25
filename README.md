# Model Comparison Plugin

A [FiftyOne plugin](https://docs.voxel51.com/plugins/index.html) for comparing two object
detection models.

## Installation

If you haven't already,
[install FiftyOne](https://docs.voxel51.com/getting_started/install.html):

```shell
pip install fiftyone
```

Then install the plugin and its dependencies:

```shell
fiftyone plugins download https://github.com/allenleetc/model-comparison-plugin
```

## Usage

1. Load your dataset. Here we create an example dataset from the [FiftyOne Dataset Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/index.html#fiftyone-dataset-zoo):

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=5000,
)

classes = ['car','person','traffic light']
three_classes = (F('label').is_in(classes))
not_crowd = (F('iscrowd')==0)
view = dataset.filter_labels('ground_truth',three_classes & not_crowd).take(200)

dataset = view.clone()
dataset.name = 'coco-simple'
dataset.persistent = True
```

2. Generate model predictions for two object detection models

```py
model_frcnn = foz.load_zoo_model('faster-rcnn-resnet50-fpn-coco-torch')
model_yolo = foz.load_zoo_model('yolov5s-coco-torch')

dataset.apply_model(model_frcnn, label_field="frcnn")
dataset.apply_model(model_yolo, label_field="yolo")
```

3. Run single-model evaluations for each model

```py
dataset.evaluate_detections('frcnn','ground_truth',eval_key='eval_frcnn',classes=classes)
dataset.evaluate_detections('yolo','ground_truth',eval_key='eval_yolo',classes=classes)
```

4. Launch the App!

```py
session = fo.launch_app(dataset)
```

5.  Press `` ` `` or click the `Browse operations` icon above the grid.

6.  Run the `Compute Model Differences` operator.

This will populate new sample- and label-level fields containing comparison statistics and metadata.

Labels are classified into groups based on whether and how they were matched in model1 compared to model2:

- hithit: model1 successful detection, model2 successful detection
- hitmiss: model1 successful detection, model2 FN
- misshit: model1 FN, model2 successful detection
- missmiss: both models FN
- hithit+: model1 and model2 successful, but localization improved in model2
- hithit-: model1 and model2 successful, but localization regressed in model2

7. Run the `View Model Differences` operator.

This operator enables viewing the groups of labels listed above with the ability to filter by class. Model improvements or regressions for across all classes, or for particular classes, can be visualized.

Tip: in the sample modal, selecting a label and using the 'z' (Crop to content) hotkey will zoom quickly to the relevant ground-truth and prediction labels.

## Implementation

FiftyOne's builtin [single-model evaluation](https://docs.voxel51.com/user_guide/evaluation.html#detections) matches ground-truth and predicted detections, storing match status (TP, FN, FP) and associated IOUs on each label.

In the `Compute Model Differences` operator, these matches are analyzed and compared across the two models. Similar to the single-model evaluation, comparison statistics are populated at the sample and label level.

The `Visualize Model Differences` operator simpifies viewing of various types of model improvements/regressions by appropriately filtering labels.

## Todo

- In `Visualize Model Differences`, if there are no samples/labels in a selected view, the entire dataset is shown.
- In `Compute Model Differences`, add the ability to specify the IOU threshold defining hithit vs hithit+ and hithit-
- Prettier icons for the operator pallete
