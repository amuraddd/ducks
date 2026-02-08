import copy
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

from ducks.models.unet import get_unnet1dmodel
from ducks.utils.training_utils import (
    load_minari_dataset, get_filtered_episodes_from_minari_dataset, 
    get_rewards_by_episode_quantile, get_dataloader_from_episodes,
    get_actions_by_episode_quantile, get_obs_by_episode_quantile
)

def train_diffusion(
    num_epochs=500,
    dataset_name="mujoco/hopper/simple-v0", 
    device="cuda"
):
    
    minari_dataset = load_minari_dataset(
        dataset_name=dataset_name
    )
    filtered_episode_datasets = get_filtered_episodes_from_minari_dataset(minari_dataset) #use these filtered episodes to create dataloaders
    
    dataloaders = get_dataloader_from_episodes(filtered_episode_datasets)
    
    #  Set the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # model
    model = get_unnet1dmodel(
        sample_size=1024,
        in_channel=11,
        out_channel=11,
        layers_per_block=2,
        block_out_channels=(32,18, 128, 256),
        device=device
    )
    models_by_episode_quantile = []
    for _ in range(len(dataloaders)):
        models_by_episode_quantile.append(copy.deepcopy(model))
    
    # return models_by_episode_quantile
    for i, (model, dataloader) in enumerate(zip(models_by_episode_quantile, dataloaders), start=1): #enumerate starts at 1 with start=1
        training_loop(
            model=model,
            model_id=i,
            dataloader=dataloader,
            noise_scheduler=noise_scheduler,
            num_epochs=500,
            device=device
        )

def training_loop(
        model,
        model_id,
        dataloader,
        noise_scheduler,
        num_epochs=500,
        device="cuda"
    ):
    # Training loop
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=4e-4
    )
    
    #add a learning rate scheduler that reduces the learning rate by a factor of 0.1 every 100 epochs
    
    
    losses = []
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            clean_samples = batch["observations"].permute(0, 2, 1).to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_samples.shape).to(clean_samples.device)
            bs = clean_samples.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_samples.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_samples, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 50 == 0:
            loss_last_epoch = sum(losses[-len(dataloader) :]) / len(dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
        
        if (epoch + 1) % (num_epochs//2) == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": sum(losses[-len(dataloader) :]) / len(dataloader) #loss at the point of checkpoint
            }
            torch.save(checkpoint, f"checkpoint_model_{model_id}.pth")