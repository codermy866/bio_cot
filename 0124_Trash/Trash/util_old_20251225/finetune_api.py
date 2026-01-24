import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

def finetune(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]) -> (str, Dict[str, float]):
    """
    通用finetune接口
    Args:
        model: nn.Module
        train_loader: DataLoader
        val_loader: DataLoader
        config: dict
    Returns:
        best_model_path: str
        metrics: dict
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4))
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    best_model_path = 'best_model.pth'
    for epoch in range(config.get('epochs', 10)):
        model.train()
        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, val_loader, config)
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), best_model_path)
    return best_model_path, val_metrics

def evaluate(model: torch.nn.Module, data_loader: DataLoader, config: Dict[str, Any]) -> Dict[str, float]:
    """
    通用evaluate接口
    Args:
        model: nn.Module
        data_loader: DataLoader
        config: dict
    Returns:
        metrics: dict
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return {'accuracy': accuracy} 