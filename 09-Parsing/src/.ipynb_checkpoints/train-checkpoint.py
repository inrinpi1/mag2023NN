from torch import nn
from torch.utils.data import DataLoader

from metrics import F1Measure, Accuracy


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    epoch_index: int,
    tb_writer
) -> float:
    # Make sure gradient tracking is on and perform training loop.
    model.train()

    epoch_loss = 0.
    running_loss = 0.

    # Use enumerate(), so that we can track batch index and do some intra-epoch reporting
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        # Show statistics at every 1000-th batch.
        if i % 1000 == 999:
            print('\t batch {} loss: {}'.format(i + 1, loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', running_loss, tb_x)
            running_loss = 0.

    return epoch_loss


def evaluate(model: nn.Module, val_dataloader: DataLoader) -> float:
    model.eval()

    lemma_scorer = F1Measure(avg='macro')
    joint_pos_feats_scorer = F1Measure(avg='macro')
    syntax_scorer = MultilabelAttachmentScore()
    misc_accuracy = F1Measure(avg='macro')
    deepslot_accuracy = F1Measure(avg='macro')
    semclass_accuracy = F1Measure(avg='macro')
    # Nulls are binary, so micro average equals to a simple binary F1.
    null_scorer = F1Measure(avg='micro')

    total_loss = 0.
    # Disable gradient computation.
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(batch)
            total_loss += outputs["loss"]
            # Average.
            mean_accuracy = np.mean([
                lemma_accuracy,
                pos_feats_accuracy,
                uas_ud,
                las_ud,
                uas_eud,
                las_eud,
                misc_accuracy,
                deepslot_accuracy,
                semclass_accuracy
            ])

    return total_loss.item()


def train(model: nn.Module, train_dataloader, val_dataloader, n_epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}')

    best_val_loss = float('inf')

    for epoch_index in range(n_epochs):
        epoch_number = epoch_index + 1
        print(f'EPOCH {epoch_number}:')

        train_loss = train_one_epoch(model, train_dataloader, epoch_index, writer)

        # Evaluate model on validation data.
        val_loss = evaluate(model, val_dataloader)

        print(f'LOSS train: {training_loss:.4f}, valid.: {val_loss:.4f}')

        # Log the running loss averaged per batch for both training and validation.
        writer.add_scalars(
            'Training vs. Validation Loss',
            {
                'Train' : train_loss,
                'Validation' : val_loss
            },
            epoch_number
        )
        writer.flush()

        # Track best performance and save model state.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model_{timestamp}_{epoch_number}"
            torch.save(model.state_dict(), model_path)

