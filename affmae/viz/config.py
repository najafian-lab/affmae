from dataclasses import dataclass, field, replace
from typing import Sequence, Tuple

__all__ = ["VizConfig", "OKABE_ITO", "PAPER", "PRESENTATION", "DEBUG"]

RGB = Tuple[float, float, float]

# Okabe-Ito: colourblind-safe and survives greyscale printing. Index 0 is black,
# reserved for background, so class c maps to entry c.
OKABE_ITO: Tuple[RGB, ...] = (
    (0.00, 0.00, 0.00),  # black      - background
    (0.90, 0.62, 0.00),  # orange
    (0.34, 0.71, 0.91),  # sky blue
    (0.00, 0.62, 0.45),  # bluish green
    (0.94, 0.89, 0.26),  # yellow
    (0.00, 0.45, 0.70),  # blue
    (0.84, 0.37, 0.00),  # vermillion
    (0.80, 0.47, 0.65),  # reddish purple
)

MAGENTA: RGB = (1.0, 0.0, 1.0)
RED: RGB = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class VizConfig:
    """How a figure should look.

    Frozen, so a config can be shared without one renderer mutating another's
    settings. Use :meth:`with_` for variants.

    Attributes:
        dpi: raster density for saved figures.
        figsize_per_cell: inches per grid cell; figures scale with their content.
        cmap: matplotlib colormap for the greyscale background image.
        palette: per-class colours. Index 0 is background. Cycles if there are
            more classes than entries.
        overlay_alpha: opacity of the class overlay on the input image.
        error_color: colour for mispredicted pixels.
        zoom_box_color: outline and leader-line colour for zoom insets.
        zoom_box_width: outline width for the zoom rectangle, in points.
        error_alpha: opacity of the error highlight.
        background_gain: contrast compression of the background image, applied
            as ``image * gain + background_lift``. Lower values push the image
            towards mid-grey so overlays read more clearly.
        background_lift: additive term of the same expression.
        token_radius: token-dot radius in pixels, or None to scale with the
            image (``max(1, img_size // 256)``), which is what makes dots
            visible at 1024 without swamping 256.
        token_color: token-dot colour, always interpreted as RGB.
        token_render_scale: supersampling factor for token figures. Dots are
            drawn at ``scale`` x resolution and downsampled, so a 1-2 px dot
            survives the figure rasterization.
        max_samples: how many samples a grid renders at most.
        font_size: base font size for titles.
        show_titles: whether to draw panel titles at all (off for paper figures
            that get captions in LaTeX).
        save_individual_cells: also write each panel as its own image file.
    """

    dpi: int = 150
    figsize_per_cell: float = 4.0
    cmap: str = "gray"
    palette: Sequence[RGB] = field(default=OKABE_ITO)
    overlay_alpha: float = 0.65
    error_color: RGB = MAGENTA
    error_alpha: float = 0.85
    background_gain: float = 0.5
    background_lift: float = 0.3
    token_radius: "int | None" = None
    token_color: RGB = RED
    token_render_scale: float = 1.0
    zoom_box_color: str = "#181A18"
    zoom_box_width: float = 4.5
    max_samples: int = 12
    font_size: int = 14
    show_titles: bool = True
    save_individual_cells: bool = False

    def __post_init__(self):
        if self.dpi < 1:
            raise ValueError(f"dpi must be >= 1, got {self.dpi}.")
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError(
                f"overlay_alpha must be in [0, 1], got {self.overlay_alpha}.")
        if not 0.0 <= self.error_alpha <= 1.0:
            raise ValueError(
                f"error_alpha must be in [0, 1], got {self.error_alpha}.")
        if self.token_render_scale < 1.0:
            raise ValueError(
                f"token_render_scale must be >= 1, got {self.token_render_scale}.")
        if self.max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {self.max_samples}.")
        if not self.palette:
            raise ValueError("palette must not be empty.")

    def with_(self, **overrides) -> "VizConfig":
        """Return a copy with fields replaced."""
        return replace(self, **overrides)

    def class_color(self, index: int) -> RGB:
        """Colour for class ``index``, cycling if the palette is shorter.

        Args:
            index: class id; 0 is background.
        Returns:
            An (r, g, b) triple in [0, 1].
        """
        return tuple(self.palette[index % len(self.palette)])

    def resolve_token_radius(self, img_size: int) -> int:
        """Token-dot radius for an image of this size.

        A fixed radius is either invisible at 1024 or covers the image at 256.
        """
        if self.token_radius is not None:
            return max(1, int(self.token_radius))
        return max(1, img_size // 256)


# Ready-made variants. `PAPER` is the default used by the training loop.
PAPER = VizConfig()
PRESENTATION = VizConfig(dpi=200, font_size=18, overlay_alpha=0.75,
                         token_render_scale=2.0)
DEBUG = VizConfig(dpi=72, max_samples=4, show_titles=True,
                  save_individual_cells=True)
