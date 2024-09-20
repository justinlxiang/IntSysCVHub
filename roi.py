from PIL import Image
from typing import Tuple
import numpy as np


class ROI:
    def __init__(self,
                 cropped_image: Image.Image,
                 top_left: Tuple[int, int],
                 bottom_right: Tuple[int, int],
                 class_prediction,
                 bitwise_mask: np.ndarray = None,
                 ) -> None:
        self._image = cropped_image
        self._top_left = top_left
        self._bottom_right = bottom_right

        # Box = (Left, Top, Right, Bottom) a.k.a (x1, y1, x2, y2)
        self._box = self._top_left + self._bottom_right

        # Pixel-level width and height of the original crop in the full image
        self._true_width = self._box[2] - self._box[0]
        self._true_height = self._box[3] - self._box[1]

        # Center-pixel of target, (x, y), in the full image
        self._center = (self._box[0] + (self._true_width // 2),
                        self._box[1] + (self._true_height // 2))

        self._image_height, self._image_width = self.np_image.shape[:2]

        # Bitwise mask (1s and 0s)
        self._mask = bitwise_mask
        if self._mask is None:
            self._mask = np.ones((self._image_height, self._image_width))

        self._verify()

        self._class_prediction = class_prediction

    # the @property function decorator is a Pythonic way to do getters, you can
    # read more about it here:
    # https://docs.python.org/3/library/functions.html?highlight=getter#property
    @property
    def class_prediction(self): return self._class_prediction

    @class_prediction.setter
    def class_prediction(self, value): self._class_prediction = value   

    @property
    def image(self): return self._image.copy()

    @property
    def np_image(self): return np.array(self.image)

    @property
    def mask(self): return self._mask.copy().astype(np.uint8)

    @property
    def masked_image(self): return Image.fromarray(self.masked_np_image)

    @property
    def masked_np_image(self): return self.np_image * \
        self.mask[:, :, np.newaxis]

    @property
    def box(self): return self._box

    @property
    def image_height(self): return self._image_height

    @property
    def image_width(self): return self._image_width

    @property
    def height(self): return self._true_height

    @property
    def width(self): return self._true_width

    @property
    def center(self): return self._center

    @property
    def top_left(self): return self._top_left

    @property
    def bottom_right(self): return self._bottom_right

    @property
    def empty_roi(self): return ROI(img, (0, 0), (1, 1), None)

    def __str__(self):
        return str(self.box)

    def _verify(self) -> None:
        assert self.width * self.height > 0, \
            f"Area of ROI is less than zero."
        assert (self.image_height, self.image_width) == self.mask.shape, \
            f"Image and mask do not have same shapes. {(self.height, self.width)} vs \
              {self.mask.shape}"
