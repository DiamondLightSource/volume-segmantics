import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_preprocess_augs(img_size: int) -> A.core.composition.Compose:

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
        ]
    )


def get_train_augs(img_size: int) -> A.core.composition.Compose:

    return A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(img_size // 2, img_size),
                height=img_size,
                width=img_size,
                p=0.5,
            ),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.04, p=0.5
                    ),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ],
                p=0.5,
            ),
            A.CLAHE(p=0.5),
            A.OneOf([A.RandomBrightnessContrast(p=0.5), A.RandomGamma(p=0.5)], p=0.5),
        ]
    )


def get_postprocess_augs() -> A.core.composition.Compose:

    return A.Compose([ToTensorV2()])
