# Model Comparison Plugin

A [FiftyOne plugin](https://docs.voxel51.com/plugins/index.html) for comparing two object
detection models.

https://github.com/allenleetc/model-comparison/assets/5833306/ba51e51b-a592-4411-8101-208a30c3713f

(Dataset: [Sama Drives California](https://huggingface.co/datasets/SamaAI/sama-drives-california))

### Installation

If you haven't already,
[install FiftyOne](https://docs.voxel51.com/getting_started/install.html):

```shell
pip install fiftyone
```

Then install the plugin and its dependencies:

```shell
fiftyone plugins download https://github.com/allenleetc/model-comparison
```

### Usage




1. Load your dataset. Here we use the [COCO-2017](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#coco-2017) from the [FiftyOne Dataset Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/index.html#fiftyone-dataset-zoo):

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=5000,
)

# Simplify dataset to three classes
classes = ['car','person','traffic light']
three_classes = (F('label').is_in(classes))
not_crowd = (F('iscrowd')==0)
view = dataset.filter_labels('ground_truth',three_classes & not_crowd).take(200)

dataset = view.clone()
dataset.name = 'coco-simple'
dataset.persistent = True
```

2. Generate model predictions using two object detection models (your dataset may already have predictions!)

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

6.  Run the `Compute Model Differences` operator. Select your ground-truth, model1 predictions, model2 predictions, and model1/model2 (single-model) evaluation keys.

This will populate new sample- and label-level fields containing statistics and metadata comparing how performance of the two models against ground truth has changed.

Labels are classified into six groups based on how they compare to ground truth for model1 compared to model2:

- **hithit**: model1 successful detection, model2 successful detection
- **hitmiss**: model1 successful detection, model2 missed detection (FN)
- **misshit**: model1 missed detection (FN), model2 successful detection
- **missmiss**: both models missed detections (FN)
- **hithit+**: model1 and model2 successful, but localization improved in model2
- **hithit-**: model1 and model2 successful, but localization regressed in model2

7. Run the `View Model Differences` operator.

This operator enables viewing the groups of labels listed above with the ability to filter by class. Model improvements or regressions across all classes, or for particular classes, can be visualized.

Tip: in the sample modal, selecting a label and using the 'z' (Crop to content) hotkey will zoom quickly to the relevant ground-truth and prediction labels.

### Implementation

FiftyOne's builtin [single-model evaluation](https://docs.voxel51.com/user_guide/evaluation.html#detections) matches ground-truth and predicted detections, storing match status (TP, FN, FP) and associated IOUs on each label.

In the `Compute Model Differences` operator, these matches are analyzed and compared across the two models. As in single-model evaluation, comparison statistics are populated at the sample and label level.

The `Visualize Model Differences` operator simpifies viewing the various types of model improvements/regressions by appropriately filtering labels.

The `Delete Model Comparison` operator deletes a model comparison run along with its sample- and label-level fields. 

Metadata for comparison runs are stored in the `dataset.info` dictionary.

### Todo

- Add tallies/counts of false positives from each prediction to include predicted detections not matched with a GT detection
- In `Visualize Model Differences`, if there are no samples/labels in a selected view, the entire dataset is shown.
- In `Compute Model Differences`, add the ability to specify the IOU threshold defining hithit vs hithit+ and hithit-
- Prettier icons for the operator pallete
