import os
import torch
import torch.nn as nn

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from torchivision import transforms as tf
from LDM_models import build_speech_to_face_pipeline, build_Lora_model
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid


@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rae = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16" # 'no' for float32, 'fp16' for automatic mixed precision
    output_dir = 'SpeechToFaceLDM'
    # push_to_hub = True 
    # hub_model_id = "<taeyang916/SpeechToFaceLDM"
    # hub_private_repo = False
    # overwrite_output_dir = True # overwrite the old model when re-running the notebook
    seed = 0

@dataclass
class ModelConfig:
    repo_id = 'ConVis/ldm-text2im-large-256'
    input_channel = 40
    output_channels = 512
    r = 8
    lora_alpha = 16
    target_modules = ['to_q', 'to_v']
    lora_dropout = 0.1
    bias = 'none'
    modules_to_save = 'unet_lora'

training_config = TrainingConfig()
model_config = ModelConfig()
training_config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(training_config.dataset_name, split='train')

image_preprocess = tf.Compose(
    [
        tf.Resize((training_config.image_size, training_config.image_size)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [image_preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_trainsform(transform)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_config.train_batch_size, shuffle=True)

s2f_pipeline = build_speech_to_face_pipeline(config=model_config)
lora_model = build_Lora_model(config=model_config, target_model=s2f_pipeline.unet)

optimizer = torch.optim.AdamW(lora_model.parameters(), lr=training_config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=training_config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * training_config.num_epochs),
)


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)