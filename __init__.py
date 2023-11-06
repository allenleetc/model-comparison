import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone as fo
from fiftyone import ViewField as F


CMP_KEY_INFO_DICT_KEY = "model_differences"
ALL_CLASSES = "<all>"


def _get_label_fields(sample_collection, label_types):
    schema = sample_collection.get_field_schema(embedded_doc_type=label_types)
    return list(schema.keys())


def save_label(label_ids_to_vals, _id, case):
    assert _id not in label_ids_to_vals
    label_ids_to_vals[_id] = case


def set_label_cmp(samples, det_fld, label_id, val, cmp_key="cmp"):
    view = samples.select_labels(ids=label_id)
    sample = view.first()
    assert len(sample[det_fld].detections) == 1
    sample[det_fld].detections[0][cmp_key] = val
    sample.save()


def analyze_match_delta(
    ctx,
    samples,
    gtfld,
    fld0,
    fld1,
    ekey0,
    ekey1,
    iou_thresh=0.2,
    cmp_key="cmp",
):
    # real dets
    # for pred0 and pred1:
    # * these are either fn (miss), or hit
    #
    # so for any det the options are
    # 1. miss -> miss
    # 2. miss -> hit  IMPROVE
    # 3. hit -> miss  REGRESS
    # 4a. hit -> hit, similar iou. SAME
    # 4b. hit -> hit, worse iou. REGRESS
    # 4c. hit -> hit, better iou. IMPROVE

    fld_match0 = ekey0
    fld_match1 = ekey1
    fld_id0 = ekey0 + "_id"
    fld_id1 = ekey1 + "_id"
    fld_iou0 = ekey0 + "_iou"
    fld_iou1 = ekey1 + "_iou"

    fld_cmp = cmp_key
    fld_cmp_id0 = cmp_key + "_id0"
    fld_cmp_id1 = cmp_key + "_id1"

    label_ids_to_vals_0 = {}
    label_ids_to_vals_1 = {}
    label_cmpfld_0 = f"{fld0}.detections.{cmp_key}"
    label_cmpfld_1 = f"{fld1}.detections.{cmp_key}"

    num_total = len(samples)
    for idx, sample in enumerate(
        samples.iter_samples(progress=True, autosave=True)
    ):
        det_map = {}
        cnt_map = {"missmiss": 0, "misshit": 0, "hitmiss": 0, "hithit": 0}
        if sample[gtfld] is not None:
            dets = sample[gtfld].detections
            for didx, d in enumerate(dets):
                _id = d["id"]
                match0 = d[fld_match0]
                match1 = d[fld_match1]
                if match0 == "fn" and match1 == "fn":
                    case = "missmiss"
                    res = (case,)
                    d[fld_cmp] = case
                elif match0 == "fn" and match1 == "tp":
                    case = "misshit"
                    id1 = d[fld_id1]
                    res = (case, id1)
                    d[fld_cmp] = case
                    d[fld_cmp_id1] = id1

                    save_label(label_ids_to_vals_1, id1, case)
                elif match0 == "tp" and match1 == "fn":
                    case = "hitmiss"
                    id0 = d[fld_id0]
                    res = (case, id0)
                    d[fld_cmp] = case
                    d[fld_cmp_id0] = id0
                    save_label(label_ids_to_vals_0, id0, case)

                else:
                    assert match0 == "tp" and match1 == "tp"
                    id0 = d[fld_id0]
                    id1 = d[fld_id1]
                    iou0 = d[fld_iou0]
                    iou1 = d[fld_iou1]

                    if iou1 - iou0 > iou_thresh:
                        case = "hithit+"
                    elif iou0 - iou1 > iou_thresh:
                        case = "hithit-"
                    else:
                        case = "hithit"

                    res = (case, id0, id1)
                    d[fld_cmp] = case
                    d[fld_cmp_id0] = id0
                    d[fld_cmp_id1] = id1

                    save_label(label_ids_to_vals_0, id0, case)
                    save_label(label_ids_to_vals_1, id1, case)

                assert _id not in det_map
                det_map[_id] = res
                case_base = case.rstrip("+-")
                cnt_map[case_base] += 1

                sample[gtfld].detections[didx] = d

            for d in sample[gtfld].detections:
                assert d[fld_cmp] is not None

        for k, v in cnt_map.items():
            k = cmp_key + "_" + k
            sample[k] = v

        progress = idx / num_total
        label = f"Loaded {idx} of {num_total}"
        yield _set_progress(ctx, progress, label=label)

    samples.set_label_values(label_cmpfld_0, label_ids_to_vals_0)
    samples.set_label_values(label_cmpfld_1, label_ids_to_vals_1)

    # Todo: false positives
    # these dont get matched. so at sample level can count

    yield


class ComputeChanges(foo.Operator):

    LABEL = "Compute Model Differences"

    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_changes",
            label=self.LABEL,
            dynamic=True,
            execute_as_generator=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _compute_changes_inputs(ctx, inputs)
        if ready:
            _execution_mode(ctx, inputs)

        return types.Property(inputs, view=types.View(label=self.LABEL))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):

        samples = ctx.dataset
        samples.reload()

        cmp_key = ctx.params.get("cmp_key", "cmp")
        gt_field = ctx.params.get("gt_field", "ground_truth")
        pd0_field = ctx.params.get("p0_field", "predictions0")
        pd1_field = ctx.params.get("p1_field", "predictions1")
        eval_key0 = ctx.params.get("ekey0", "evaluation0")
        eval_key1 = ctx.params.get("ekey1", "evlauation1")

        info_dict = samples.info
        if CMP_KEY_INFO_DICT_KEY not in info_dict:
            info_dict[CMP_KEY_INFO_DICT_KEY] = {}

        info_dict[CMP_KEY_INFO_DICT_KEY][cmp_key] = {
            "gt_field": gt_field,
            "pd0_field": pd0_field,
            "pd1_field": pd1_field,
            "eval_key0": eval_key0,
            "eval_key1": eval_key1,
        }
        samples.save()

        for update in analyze_match_delta(
            ctx,
            samples,
            gt_field,
            pd0_field,
            pd1_field,
            eval_key0,
            eval_key1,
            iou_thresh=0.1,
            cmp_key=cmp_key,
        ):
            yield update

        yield samples.add_dynamic_sample_fields()
        yield ctx.trigger("reload_dataset")


def add_menu(ctx, inputs, input_name, choices_view, label, description=None):
    inputs.enum(
        input_name,
        choices_view.values(),
        required=True,
        label=label,
        description=description,
        view=choices_view,
    )
    input_val = ctx.params.get(input_name, None)
    return input_val is not None


def _compute_changes_inputs(ctx, inputs):

    dataset = ctx.dataset
    label_fields = _get_label_fields(dataset, (fo.Detections,))
    label_field_choices = types.DropdownView()
    for field_name in sorted(label_fields):
        label_field_choices.add_choice(field_name, label=field_name)

    if not add_menu(
        ctx,
        inputs,
        "gt_field",
        label_field_choices,
        "Ground Truth Field",
        "The label field containing ground truth detections",
    ):
        return False

    if not add_menu(
        ctx,
        inputs,
        "p0_field",
        label_field_choices,
        "First Model Field",
        "The label field containing the first model predictions",
    ):
        return False

    if not add_menu(
        ctx,
        inputs,
        "p1_field",
        label_field_choices,
        "Second Model Field",
        "The label field containing the second model predictions",
    ):
        return False

    evals = dataset.list_evaluations()
    eval_field_choices = types.DropdownView()
    for eval in sorted(evals):
        eval_field_choices.add_choice(eval, label=eval)

    if not add_menu(
        ctx,
        inputs,
        "ekey0",
        eval_field_choices,
        "Evaluation Key for First Model",
    ):
        return False

    if not add_menu(
        ctx,
        inputs,
        "ekey1",
        eval_field_choices,
        "Evaluation Key for Second Model",
    ):
        return False

    inputs.str(
        "cmp_key",
        required=True,
        label="Model Comparison Key",
        description="Supply a key for this model comparison",
    )

    cmp_key = ctx.params.get("cmp_key", None)
    if cmp_key is None:
        return False

    return types.Property(
        inputs, view=types.View(label="Compute Model Differences")
    )


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations "
                    "for more information"
                )
            ),
        )


def _set_progress(ctx, progress, label=None):
    # https://github.com/voxel51/fiftyone/pull/3516
    # return ctx.trigger("set_progress", dict(progress=progress, label=label))

    loading = types.Object()
    loading.float("progress", view=types.ProgressView(label=label))
    return ctx.trigger(
        "show_output",
        dict(
            outputs=types.Property(loading).to_json(),
            results={"progress": progress},
        ),
    )


def _view_changes_menu(ctx, inputs):

    dataset = ctx.dataset
    info_dict = dataset.info
    if CMP_KEY_INFO_DICT_KEY not in info_dict:
        return False

    label_fields = list(info_dict[CMP_KEY_INFO_DICT_KEY].keys())
    label_field_choices = types.DropdownView()
    for field_name in sorted(label_fields):
        label_field_choices.add_choice(field_name, label=field_name)

    if not add_menu(
        ctx,
        inputs,
        "cmp_key",
        label_field_choices,
        "Comparison Key",
        "The comparison key used when Computing Model Differences",
    ):
        return False

    cmp_key = ctx.params.get("cmp_key", None)
    cmp_dict = dataset.info[CMP_KEY_INFO_DICT_KEY][cmp_key]
    gt_field = cmp_dict["gt_field"]
    gt_det_cmp_field = f"{gt_field}.detections.{cmp_key}"
    type_vals = dataset.distinct(gt_det_cmp_field)
    type_selector = types.AutocompleteView()
    for ty in type_vals:
        type_selector.add_choice(ty, label=ty)

    if not add_menu(
        ctx,
        inputs,
        "type",
        type_selector,
        "Type of change to view",  # "type desc"
    ):
        return False

    gt_cls_field = f"{gt_field}.detections.label"
    gt_classes = dataset.distinct(gt_cls_field)
    gt_classes = [
        ALL_CLASSES,
    ] + gt_classes
    class_selector = types.AutocompleteView()
    for cl in gt_classes:
        class_selector.add_choice(cl, label=cl)

    if not add_menu(
        ctx,
        inputs,
        "class",
        class_selector,
        "Class to view",
    ):
        return False

    return types.Property(
        inputs, view=types.View(label="View  Model Differences")
    )


def _delete_comparison_menu(ctx, inputs):

    dataset = ctx.dataset
    info_dict = dataset.info
    if CMP_KEY_INFO_DICT_KEY not in info_dict:
        return False

    label_fields = list(info_dict[CMP_KEY_INFO_DICT_KEY].keys())
    label_field_choices = types.DropdownView()
    for field_name in sorted(label_fields):
        label_field_choices.add_choice(field_name, label=field_name)

    if not add_menu(
        ctx, inputs, "cmp_key", label_field_choices, "Comparison Key"
    ):
        return False

    return types.Property(
        inputs, view=types.View(label="View  Model Differences")
    )


class ViewChanges(foo.Operator):

    LABEL = "View Model Differences"

    @property
    def config(self):
        return foo.OperatorConfig(
            name="view_changes",
            label=ViewChanges.LABEL,
            dynamic=True,  #            execute_as_generator=True,
        )

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _view_changes_menu(ctx, inputs)
        # if ready:
        #    inputs.bool(
        #        "groupby_scene",
        #        default=False,
        #        required=True,
        #        label="Group by scene?",
        #        view=types.CheckboxView(),
        #    )
        #    groupby_scene = ctx.params.get("groupby_scene", False)

        return types.Property(inputs, view=types.View(label=self.LABEL))

    def execute(self, ctx):
        dataset = ctx.dataset

        cmp_key = ctx.params.get("cmp_key", None)
        type = ctx.params.get("type", None)
        label_class = ctx.params.get("class", None)
        cmp_dict = dataset.info[CMP_KEY_INFO_DICT_KEY][cmp_key]
        gt_field = cmp_dict["gt_field"]
        pd0_field = cmp_dict["pd0_field"]
        pd1_field = cmp_dict["pd1_field"]
        if label_class == ALL_CLASSES:
            view_expr = F(cmp_key) == type
        else:
            view_expr = (F(cmp_key) == type) & (F("label") == label_class)

        if type == "misshit":
            view = (
                dataset.filter_labels(gt_field, view_expr)
                .filter_labels(pd1_field, view_expr)
                .filter_labels(pd0_field, view_expr, only_matches=False)
            )
        elif type == "hitmiss":
            view = (
                dataset.filter_labels(gt_field, view_expr)
                .filter_labels(pd0_field, view_expr)
                .filter_labels(pd1_field, view_expr, only_matches=False)
            )
        elif type == "missmiss":
            view = (
                dataset.filter_labels(gt_field, view_expr)
                .filter_labels(pd0_field, view_expr, only_matches=False)
                .filter_labels(pd1_field, view_expr, only_matches=False)
            )

        else:
            view = (
                dataset.filter_labels(gt_field, view_expr)
                .filter_labels(pd0_field, view_expr)
                .filter_labels(pd1_field, view_expr)
            )

        # groupby_scene = ctx.params.get('groupby_scene',None)
        # if groupby_scene:
        #    view = view.group_by('scene',order_by='frame')

        ctx.trigger("set_view", {"view": view._serialize()})

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("updated", label="Updated")
        return types.Property(outputs)


class DeleteComparison(foo.Operator):

    LABEL = "Delete Model Comparison"

    @property
    def config(self):
        return foo.OperatorConfig(
            name="delete_comparison",
            label=DeleteComparison.LABEL,
            dynamic=True,
        )

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _delete_comparison_menu(ctx, inputs)

        return types.Property(inputs, view=types.View(label=self.LABEL))

    def execute(self, ctx):
        dataset = ctx.dataset

        fields = dataset.get_field_schema(flat=True).keys()
        cmp_key = ctx.params.get("cmp_key", None)
        fields_rm = [
            x for x in fields if cmp_key in x
        ]  # Assumes unique-ish cmp_key!

        dataset.delete_sample_fields(fields_rm)

        info_dict = dataset.info
        info_dict[CMP_KEY_INFO_DICT_KEY].pop(cmp_key, None)

        ctx.trigger("reload_dataset")

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("Success", label="Success")
        return types.Property(outputs)


def register(p):
    p.register(ViewChanges)
    p.register(ComputeChanges)
    p.register(DeleteComparison)
