from datetime import datetime
from loaders import *
from utilities import Logger

if __name__ == '__main__':
    device = getDevice()
    print('device is:',device)
    log_path = os.path.join(sys.path[0], 'log_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt')
    print('log_path is:',log_path)

    print('loading tokenizer...')
    tokenizer = getTokenizer()
    print('loading model...')
    model = getModel()
    print('getting data...')
    dataloader = getDataloader(texts_tr, labels_tr, tokenizer, max_length=512, batch_size=32)
    dataloader_vl = getDataloader(texts_vl, labels_vl, tokenizer, max_length=512, batch_size=32)
    print('model to device...')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.functional.cross_entropy

    start_time = datetime.now()
    logger = Logger(log_path)
    logger.add("Training and Validation -> Start Time: " + start_time.strftime("H%M%S"))

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        logger.add(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        loss,acc = compute_epoch(model, dataloader, optimizer, criterion, epoch=epoch, device=device)
        logger.add(f'Epoch: {epoch}, TR Loss: {loss:.4f}, TR accuracy: {acc:.2f}')
        model.eval()
        loss,acc = compute_epoch(model, dataloader_vl, optimizer, criterion, backpropagate=False, epoch=epoch, device=device)
        logger.add(f'Epoch: {epoch}, VL Loss: {loss:.4f}, VL accuracy: {acc:.2f}')