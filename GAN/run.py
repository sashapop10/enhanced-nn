import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
import model


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained generator model
# Epoch 110 Iteration 100: discriminator_loss 0.682 generator_loss 0.731
G = torch.load("Generator_epoch_109.pth", map_location=device)
G.eval()  # Set the model to evaluation mode

# Generate and save images
num_images = 10  # Number of images to generate
z_dim = 128  # Dimension of the noise vector

# Generate random noise vectors
with torch.no_grad():
    noise = torch.randn(num_images, z_dim).to(device)

# Generate fake images
fake_images = G(noise)

# Denormalize and save images
fake_images = (fake_images + 1) / 2  # Denormalize the images
save_image(fake_images, "generated_images.png", nrow=5)  # Save the images in a grid
