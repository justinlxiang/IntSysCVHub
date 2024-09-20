import logging

from detectron2.data import transforms as T

def build_transform(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing, horizontal flipping, and vertical flipping.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        tfm_gens.append(T.RandomFlip())
        tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens
