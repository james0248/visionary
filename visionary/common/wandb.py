from omegaconf import DictConfig, OmegaConf

import wandb


class WandbLogger:
    def __init__(self, cfg: DictConfig):
        self.enabled = cfg.wandb.enabled
        if self.enabled:
            wandb.init(
                project=cfg.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    def log(self, data: dict, step: int):
        if self.enabled:
            wandb.log(data, step=step)

    def log_video(self, key: str, video_path: str, step: int):
        if self.enabled:
            wandb.log({key: wandb.Video(video_path)}, step=step)

    def log_image(self, key: str, image, step: int, caption: str | None = None):
        if self.enabled:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()
