import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import model_context
from src.model_context import ModelContext
import tqdm as tqdm


def custom_mse_loss(output, target, weights, model_context: ModelContext, epsilon=1e-6):
    loss = ((output - target) ** 2)
    loss = (loss * weights).sum() / (weights.sum() + epsilon)
    return loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, train_iters, save_model_path, model_context: ModelContext, 
                custom_loss=False,
                coord_weights=(1.0, 3.0), 
                time_decay=1.0):
    avg_losses = []
    avg_val_losses = []

    device = model_context.device
    data_type = model_context.data_type

    min_val_loss = float('inf')

    weights = None 

    if custom_loss:
        cw = torch.tensor(coord_weights, dtype=model_context.data_type).to(model_context.device).view(1, 1, 2)
        td = (torch.tensor(time_decay, dtype=model_context.data_type).to(model_context.device) 
            ** torch.arange(model_context.waypoint_num, dtype=model_context.data_type).to(model_context.device)).view(1, model_context.waypoint_num, 1)
        weights = cw * td
        

    for epoch in range(train_iters):
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{train_iters}', unit='batch')

        for i, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device).to(data_type)
            targets = targets.to(device).to(data_type)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = None
            if custom_loss:
                loss = custom_mse_loss(outputs, targets, weights, model_context)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{train_iters}], Loss: {avg_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device).to(data_type)
                targets = targets.to(device).to(data_type)

                outputs = model(inputs)
                loss = None
                if custom_loss:
                    loss = custom_mse_loss(outputs, targets, weights, model_context)
                else:
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        base_path = '/Users/tom/Development/carbot_ai_model/models'

        if (avg_val_loss < min_val_loss):
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{base_path}/{save_model_path}/best_model.pth')
            print(f'Saved best model with validation loss: {min_val_loss:.4f}')

        avg_losses.append(avg_loss)
        avg_val_losses.append(avg_val_loss)

    return avg_losses, avg_val_losses

def view_val_train_loss(avg_losses, avg_val_losses):
    plt.plot(avg_losses, label='Training Loss')
    plt.plot(avg_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()



def view_evaluation_sample(o_t, waypoints, model, model_context: ModelContext):
    model.eval()
    with torch.no_grad():
        input_tensor = o_t.unsqueeze(0).to(model_context.device).to(model_context.data_type)  # Add batch dimension
        predicted_waypoints = model(input_tensor).squeeze(0)  # Remove batch dimension

    # Denormalize and convert image for visualization

    if (len(o_t.shape) == 4):
        o_t = o_t[0]
    
    print(o_t.shape)

    o_t_cpu = o_t.to('cpu')
    for t, m, s in zip(o_t_cpu, model_context.mean, model_context.std):
        t.mul_(s).add_(m)
    o_t_cpu = o_t_cpu.permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_title('Input Image')
    axs[0].imshow(o_t_cpu)
    axs[0].axis('off')

    axs[1].set_title('Waypoints Visualization')


    xy = waypoints[:, :2]
    xy_pred = predicted_waypoints[:, :2]
    x = xy[:, 0].cpu().numpy()
    y = xy[:, 1].cpu().numpy() * -1
    x_pred = xy_pred[:, 0].cpu().numpy()
    y_pred = xy_pred[:, 1].cpu().numpy() * -1

    max_y = max(abs(max(y)), abs(min(y)))
    max_y_pred = max(abs(max(y_pred)), abs(min(y_pred)))
    max_y = max(max_y, max_y_pred) + 0.5

    axs[1].set_xlim(-max_y, max_y)

    axs[1].set_xlabel('Y (m)')
    axs[1].set_ylabel('X (m)')
    axs[1].legend(['Ground Truth', 'Predicted'])

    axs[1].plot(y, x, marker='o')
    axs[1].plot(y_pred, x_pred, marker='x')

    # Label points 1–8
    for i in range(model_context.waypoint_num):
        axs[1].text(y[i], x[i], str(i + 1), fontsize=12, ha='right', va='bottom')

    for i in range(model_context.waypoint_num):
        axs[1].text(y_pred[i], x_pred[i], str(i + 1), fontsize=12, ha='left', va='top', color='red')

    axs[1].grid()
    plt.show()