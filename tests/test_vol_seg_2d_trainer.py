from volume_segmantics.model import VolSeg2dTrainer
import pytest
import torch


@pytest.fixture()
def volseg_2d_trainer(image_dir, label_dir, training_settings):
    return VolSeg2dTrainer(image_dir, label_dir, 4, training_settings)


def find_frozen_params(model):
    param_list = []
    for name, param in model.named_parameters():
        if all(["encoder" in name, "conv" in name]) and not param.requires_grad:
            param_list.append(name)
    return param_list


class TestVolSeg2dTrainer:
    @pytest.mark.gpu
    def test_2d_trainer_init(self, volseg_2d_trainer):
        assert isinstance(volseg_2d_trainer, VolSeg2dTrainer)
        assert volseg_2d_trainer.label_no == 4
        assert volseg_2d_trainer.codes == {}
        assert isinstance(volseg_2d_trainer.loss_criterion, torch.nn.Module)

    @pytest.mark.gpu
    def test_create_model_and_optimiser(self, volseg_2d_trainer):
        volseg_2d_trainer.create_model_and_optimiser(learning_rate=0.001, frozen=True)
        assert isinstance(volseg_2d_trainer.model, torch.nn.Module)
        device = next(volseg_2d_trainer.model.parameters()).device
        assert device.type == "cuda"
        assert device.index == 0
        assert isinstance(volseg_2d_trainer.optimizer, torch.optim.AdamW)

    @pytest.mark.gpu
    def test_unfreeze_model(self, volseg_2d_trainer):
        volseg_2d_trainer.create_model_and_optimiser(learning_rate=0.001, frozen=True)
        assert isinstance(volseg_2d_trainer.model, torch.nn.Module)
        param_list = find_frozen_params(volseg_2d_trainer.model)
        assert len(param_list) > 0
        volseg_2d_trainer.unfreeze_model()
        param_list = find_frozen_params(volseg_2d_trainer.model)
        assert len(param_list) == 0

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "loss_name",
        [
            "BCEDiceLoss",
            "DiceLoss",
            "BCELoss",
            "CrossEntropyLoss",
            "GeneralizedDiceLoss",
        ],
    )
    def test_get_loss_criterion(self, volseg_2d_trainer, loss_name):
        volseg_2d_trainer.settings.loss_criterion = loss_name
        criterion = volseg_2d_trainer.get_loss_criterion()
        assert isinstance(criterion, torch.nn.Module)

    @pytest.mark.gpu
    def test_get_loss_criterion_bad_loss(
        self, volseg_2d_trainer, loss_name="lossnessmonster"
    ):
        volseg_2d_trainer.settings.loss_criterion = loss_name
        with pytest.raises(SystemExit) as wrapped_e:
            volseg_2d_trainer.get_loss_criterion()
        assert wrapped_e.type == SystemExit
        assert wrapped_e.value.code == 1

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "eval_metric_name",
        [
            "MeanIoU",
            "GenericAveragePrecision",
        ],
    )
    def test_get_eval_metric(self, volseg_2d_trainer, eval_metric_name):
        volseg_2d_trainer.settings.eval_metric = eval_metric_name
        metric = volseg_2d_trainer.get_eval_metric()
        assert hasattr(metric, "__dict__")

    @pytest.mark.gpu
    def test_get_eval_metric_bad_metric(
        self, volseg_2d_trainer, eval_metric_name="evaluatethis"
    ):
        volseg_2d_trainer.settings.eval_metric = eval_metric_name
        with pytest.raises(SystemExit) as wrapped_e:
            metric = volseg_2d_trainer.get_eval_metric()
        assert wrapped_e.type == SystemExit
        assert wrapped_e.value.code == 1

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_train_new_frozen_model(self, volseg_2d_trainer, empty_dir):
        output_path = empty_dir / "my_model.pytorch"
        volseg_2d_trainer.train_model(
            output_path, num_epochs=1, patience=1, create=True, frozen=True
        )
        assert output_path.is_file()
        volseg_2d_trainer.output_loss_fig(output_path)
        loss_fig_path = empty_dir / "my_model_loss_plot.png"
        assert loss_fig_path.is_file()
        volseg_2d_trainer.output_prediction_figure(output_path)
        pred_fig_path = empty_dir / "my_model_prediction_image.png"
        assert pred_fig_path.is_file()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_train_existing_frozen_model(self, volseg_2d_trainer, model_path):
        volseg_2d_trainer.train_model(
            model_path, num_epochs=1, patience=1, create=False, frozen=True
        )
        assert model_path.is_file()
