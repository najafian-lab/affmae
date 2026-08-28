"""Released checkpoints for the EM / FPW dataset, and how to fetch them.

Each entry records what the checkpoint *is* -- backbone, stage, resolution, class
count, the config it was trained with -- next to its download link, so nothing
has to be inferred from a filename. Pass one straight to the loader:

    from affmae import AFFMAE
    from affmae.data.weights import EMWeights

    model = AFFMAE.from_checkpoint(EMWeights.AFFMAE_BASE_FT_512)

The file is downloaded once into the local weights cache and reused after that.
An explicit URL works too, for checkpoints that are not in this registry.

Fetching needs ``requests``; it is not a hard dependency, and the error names the
browser URL so a manual download is always possible.
"""

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["CheckpointSpec", "EMWeights", "download_file", "resolve_source",
           "WEIGHTS_FOLDER_URL"]

#: The public Drive folder holding every file below.
WEIGHTS_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1ZGnBMpduV43wgiTVMJCyBSYUiMRtM_NW")

_VIEW = "https://drive.google.com/file/d/{file_id}/view"
#: Cookieless direct-download endpoint. ``confirm=t`` skips the
#: "file is too large to scan for viruses" interstitial that the older
#: ``/uc?export=download`` path returns as HTML for anything over ~100 MB.
_DIRECT = ("https://drive.usercontent.google.com/download"
           "?id={file_id}&export=download&confirm=t")


@dataclass(frozen=True)
class CheckpointSpec:
    """One released checkpoint.

    Attributes:
        filename: name to cache it under, matching the file in the Drive folder.
        backbone: ``"affmae"`` or ``"vit"``.
        task: ``"pretrain"`` for an MAE checkpoint, ``"segmentation"`` for a
            finetuned one. This is what :meth:`AFFMAE.from_checkpoint` needs to
            know which head to build.
        img_size: resolution it was trained at. A finetuned checkpoint is
            resolution-specific: the token count scales with img_size/patch_size.
        patch_size: encoder patch size.
        num_classes: including background; None for a pretraining checkpoint.
        config: the config in ``configs/`` it was trained with.
        gdrive_id: Drive file id.
        description: one line, for the README table and ``--help`` output.
    """

    filename: str
    backbone: str
    task: str
    img_size: int
    patch_size: int
    num_classes: Optional[int]
    config: str
    gdrive_id: str
    description: str

    @property
    def url(self) -> str:
        """Human-facing Drive link, for a browser or a README table."""
        return _VIEW.format(file_id=self.gdrive_id)

    @property
    def download_url(self) -> str:
        """Direct link that streams bytes rather than an HTML preview."""
        return _DIRECT.format(file_id=self.gdrive_id)


class EMWeights(Enum):
    """Released checkpoints, keyed by backbone, stage and resolution.

    Members carry a :class:`CheckpointSpec`. The properties forward to it, so
    ``EMWeights.AFFMAE_BASE_FT_512.img_size`` works without reaching for
    ``.value``.
    """

    AFFMAE_BASE_PRETRAIN_512 = CheckpointSpec(
        filename="ckpt_epoch_399_affmae_fpw.pth",
        backbone="affmae", task="pretrain", img_size=512, patch_size=8,
        num_classes=None,
        config="configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml",
        gdrive_id="1-2dkpOv4Q6f3jrX3Lom02wnCMasq9TbS",
        description="AFF-MAE Base, 400 epochs of masked autoencoding on the EM corpus.")

    AFFMAE_BASE_FT_512 = CheckpointSpec(
        filename="fpw_aff_base_ft_512_slits_pgbmi.pth",
        backbone="affmae", task="segmentation", img_size=512, patch_size=8,
        num_classes=3, config="configs/aff_base_finetune_512_fpw.yaml",
        gdrive_id="13TcyZG9Gd-0vxkAFXpe0oRtaAvVsWXTq",
        description="AFF-MAE Base finetuned on FPW at 512px; background, PGBMI, filtration slits.")

    AFFMAE_BASE_FT_768 = CheckpointSpec(
        filename="fpw_aff_base_ft_768_slits_pgbmi.pth",
        backbone="affmae", task="segmentation", img_size=768, patch_size=8,
        num_classes=3, config="configs/aff_base_finetune_768.yaml",
        gdrive_id="17YqfGYbXduXgAC14frn9gNCuMJ7-653Y",
        description="AFF-MAE Base finetuned on FPW at 768px.")

    AFFMAE_BASE_FT_1024 = CheckpointSpec(
        filename="fpw_aff_base_ft_1024_slits_pgbmi.pth",
        backbone="affmae", task="segmentation", img_size=1024, patch_size=8,
        num_classes=3, config="configs/aff_base_finetune_1024_fpw.yaml",
        gdrive_id="1YDlfN1Gm5qBv8WhIdrL9ujeQamDqXUS_",
        description="AFF-MAE Base finetuned on FPW at 1024px.")

    VIT_BASE_PRETRAIN_512 = CheckpointSpec(
        filename="ckpt_epoch_399_vit_base.pth",
        backbone="vit", task="pretrain", img_size=512, patch_size=16,
        num_classes=None, config="configs/vit_base_pretrain_0.5mask.yaml",
        gdrive_id="1ZYsiE_GxBDmunx0VCGhfOh4bP9fwWKhg",
        description="ViT-Base MAE baseline, same corpus and schedule as the AFF-MAE pretrain.")

    VIT_BASE_FT_512 = CheckpointSpec(
        filename="fpw_vit_ft_512_slits_pgbmi.pth",
        backbone="vit", task="segmentation", img_size=512, patch_size=16,
        num_classes=3, config="configs/vit_base_finetune_fpn_512.yaml",
        gdrive_id="1A7spm6E7Rbq56LNzlgdUCuIhPN4-bT7-",
        description="ViT-Base + UperNet baseline finetuned on FPW at 512px.")

    VIT_BASE_FT_768 = CheckpointSpec(
        filename="fpw_vit_ft_768_slits_pgbmi.pth",
        backbone="vit", task="segmentation", img_size=768, patch_size=16,
        num_classes=3, config="configs/vit_base_finetune_fpn_768.yaml",
        gdrive_id="1zPbites1OuYJsXEYzKkElmXsBxMUaFi3",
        description="ViT-Base + UperNet baseline finetuned on FPW at 768px.")

    VIT_BASE_FT_1024 = CheckpointSpec(
        filename="fpw_vit_ft_1024_slits_pgbmi.pth",
        backbone="vit", task="segmentation", img_size=1024, patch_size=16,
        num_classes=3, config="configs/vit_base_finetune_fpn_1024.yaml",
        gdrive_id="1R3lj9kBFl0jAd_O1H9mwaGpik2vgo-_-",
        description="ViT-Base + UperNet baseline finetuned on FPW at 1024px.")

    # -- forwarding, so callers never touch .value --------------------------- #

    @property
    def spec(self) -> CheckpointSpec:
        return self.value

    @property
    def filename(self) -> str:
        return self.value.filename

    @property
    def backbone(self) -> str:
        return self.value.backbone

    @property
    def task(self) -> str:
        return self.value.task

    @property
    def img_size(self) -> int:
        return self.value.img_size

    @property
    def patch_size(self) -> int:
        return self.value.patch_size

    @property
    def num_classes(self) -> Optional[int]:
        return self.value.num_classes

    @property
    def config(self) -> str:
        return self.value.config

    @property
    def url(self) -> str:
        return self.value.url

    @property
    def download_url(self) -> str:
        return self.value.download_url

    @property
    def description(self) -> str:
        return self.value.description

    @property
    def download_path(self) -> str:
        """Where this checkpoint is cached, without downloading it.

        Defaults into the project's own weight folders -- ``weights/pretrain/``
        for MAE checkpoints and ``weights/segmentation/`` for finetuned ones,
        the layout ``weights/README.md`` documents. Set ``CHECKPOINT_ROOT`` to
        relocate the whole tree.
        """
        from affmae.utils.paths import repo_root

        root = os.environ.get("CHECKPOINT_ROOT") or os.path.join(
            str(repo_root()), "weights")
        return os.path.join(root, self.task, self.filename)

    def local_path(self) -> str:
        """Deprecated alias for :attr:`download_path`."""
        return self.download_path

    def fetch(self, dest: Optional[str] = None, force: bool = False,
              progress: bool = True) -> str:
        """Download this checkpoint if it is not already cached.

        Args:
            dest: explicit destination path; defaults to :attr:`download_path`.
            force: re-download even if the file is present.
            progress: show a tqdm bar.
        Returns:
            Path to the local file.
        """
        target = dest or self.download_path
        return download_file(self.download_url, target, force=force,
                             progress=progress, label=self.filename)


def download_file(url: str, dest: str, force: bool = False,
                  progress: bool = True, label: Optional[str] = None) -> str:
    """Stream ``url`` to ``dest``, with a progress bar.

    Downloads to a ``.part`` file and moves it into place only on success, so an
    interrupted transfer cannot leave a truncated checkpoint that later fails to
    unpickle with a confusing error.

    Args:
        url: direct download URL.
        dest: destination path; parent directories are created.
        force: overwrite an existing file.
        progress: show a tqdm bar when tqdm is installed.
        label: name shown in the bar; defaults to the destination filename.
    Returns:
        ``dest``.
    Raises:
        RuntimeError: if ``requests`` is missing, the response is an error, or
            Drive returned its HTML interstitial instead of the file.
    """
    if os.path.exists(dest) and not force:
        return dest

    try:
        import requests
    except ImportError as error:
        raise RuntimeError(
            f"downloading needs `requests` (pip install requests). Or fetch "
            f"the file manually and pass its path:\n  {url}") from error

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    partial = dest + ".part"

    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"download failed with HTTP {response.status_code} for {url}")
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            # Drive serves an HTML confirmation page when a link is not public
            # or the confirm token is missing; writing it would produce a
            # plausible-looking file that fails much later, at torch.load.
            raise RuntimeError(
                f"expected a file but got HTML from {url}. The link may not be "
                f"publicly shared. Download it in a browser and pass the path.")

        total = int(response.headers.get("Content-Length") or 0)
        bar = None
        if progress:
            try:
                from tqdm import tqdm

                bar = tqdm(total=total or None, unit="B", unit_scale=True,
                           unit_divisor=1024,
                           desc=label or os.path.basename(dest))
            except ImportError:
                print(f"downloading {label or os.path.basename(dest)} "
                      f"({total / 1e6:.0f} MB)...")

        try:
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        finally:
            if bar is not None:
                bar.close()

    written = os.path.getsize(partial)
    if total and written != total:
        os.remove(partial)
        raise RuntimeError(
            f"truncated download: got {written} bytes, expected {total}. "
            f"Retry, or download manually from {url}")

    shutil.move(partial, dest)
    return dest


def _lookup_member(text: str):
    """Return the EMWeights member ``text`` names, or None if it is not a name.

    Only strings shaped like a member name are considered, so a real file called
    ``AFFMAE_BASE_FT_512`` in the current directory still wins: a name has no
    path separator, no file suffix, and is upper case.
    """
    if os.sep in text or "/" in text or "." in text:
        return None
    if not text or text != text.upper():
        return None
    if os.path.exists(text):
        return None
    try:
        return EMWeights[text]
    except KeyError:
        raise KeyError(
            f"{text!r} is not a released checkpoint. Available: "
            f"{', '.join(m.name for m in EMWeights)}. Pass a path or URL "
            f"instead if you meant a local file.") from None


def resolve_source(source, progress: bool = True):
    """Turn a checkpoint reference into a local path, downloading if needed.

    Accepts what a user is likely to have: an :class:`EMWeights` member or its
    name, an ``http(s)`` URL, a Drive share link, or a path that already exists.

    The name form is what makes the registry reachable from a command line,
    where every argument is a string: ``--checkpoint AFFMAE_BASE_FT_512``.

    Args:
        source: EMWeights member, member name, URL, or filesystem path.
        progress: show a progress bar for downloads.
    Returns:
        ``(path, spec)``. ``spec`` is the CheckpointSpec when the source named a
        registry entry, else None -- the caller then has no metadata and must
        fall back to the config.
    Raises:
        KeyError: if the source looks like a registry name (no path separator,
            no suffix, all upper case) but does not match an entry. Guessing a
            filesystem path from a near-miss like ``AFFMAE_BASE_FT_513`` would
            surface as a confusing FileNotFoundError instead.
    """
    if isinstance(source, EMWeights):
        return source.fetch(progress=progress), source.spec

    text = str(source)
    member = _lookup_member(text)
    if member is not None:
        return member.fetch(progress=progress), member.spec

    if text.startswith(("http://", "https://")):
        url, filename = _normalize_url(text)
        from affmae.utils.paths import repo_root

        root = os.environ.get("CHECKPOINT_ROOT") or os.path.join(
            str(repo_root()), "weights")
        dest = os.path.join(root, "downloaded", filename)
        return download_file(url, dest, progress=progress, label=filename), None

    return text, None


def _normalize_url(url: str):
    """Rewrite a Drive share link into a direct download, and pick a filename.

    ``https://drive.google.com/file/d/<id>/view`` is a preview page, not the
    file; passing it to a downloader yields HTML.
    """
    import re
    from urllib.parse import parse_qs, urlparse

    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if match is None and "drive.google.com" in url:
        ids = parse_qs(urlparse(url).query).get("id")
        match = ids[0] if ids else None
        file_id = match
    else:
        file_id = match.group(1) if match else None

    if file_id:
        known = {entry.spec.gdrive_id: entry.filename for entry in EMWeights}
        return _DIRECT.format(file_id=file_id), known.get(file_id, f"{file_id}.pth")

    name = os.path.basename(urlparse(url).path) or "checkpoint.pth"
    return url, name
